"""
训练器模块
"""
import os
import torch
import logging
import shutil
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from transformers import get_scheduler
import swanlab

from config import TrainingConfig
from model import ImageTextEncoder

logger = logging.getLogger(__name__)

class Trainer:
    """
    负责模型训练和评估的训练器。
    """
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, scheduler, model_config, data_config, training_config, swanlab_config, processor=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.swanlab_config = swanlab_config
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化 SwanLab
        if self.swanlab_config.enable:
            # 合并所有配置用于记录
            config_dict = {
                **model_config.__dict__,
                **data_config.__dict__,
                **training_config.__dict__
            }
            
            swanlab.init(
                project=self.swanlab_config.project,
                experiment_name=self.swanlab_config.experiment_name,
                description=self.swanlab_config.description,
                config=config_dict,
                mode=self.swanlab_config.mode
            )
            logger.info("SwanLab initialized successfully")
        
        # Loss 权重配置（可在此处修改）
        # 激进策略: 降低 Random 权重，大幅提升 Explicit Hard 权重
        self.random_negative_weight = 0.5  # 保持基础区分度
        self.explicit_negative_weight = 2.0  # 重点强调区分细节错误
        self.other_negative_weight = 0.5    # 充分利用其余负样本

        # 创建唯一的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(training_config.output_dir, f"training_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存完整配置信息到 config.txt
        with open(os.path.join(self.output_dir, "config.txt"), "w", encoding="utf-8") as f:
            f.write("=== Model Config ===\n")
            f.write(str(model_config))
            f.write("\n\n=== Data Config ===\n")
            f.write(str(data_config))
            f.write("\n\n=== Training Config ===\n")
            f.write(str(training_config))
            f.write("\n\n=== Loss Configuration ===\n")
            f.write(f"Loss = {self.random_negative_weight} * Random_Loss + {self.explicit_negative_weight} * Explicit_Hard_Loss + {self.other_negative_weight} * Other_Negative_Loss\n")
            f.write(f"Random Negative Weight: {self.random_negative_weight}\n")
            f.write(f"Explicit Negative Weight: {self.explicit_negative_weight}\n")
            f.write(f"Other Negative Weight: {self.other_negative_weight}\n")
        # 配置日志文件
        file_handler = logging.FileHandler(os.path.join(self.output_dir, "training.log"), encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        # 同时给root logger添加，确保所有模块的日志都能写入文件
        logging.getLogger().addHandler(file_handler)
        
        self.best_accuracy = 0.0
        self.best_model_dir = None  # 记录当前最佳模型目录
        
        logger.info(f"训练器初始化完成，可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        logger.info(f"训练输出将保存在: {self.output_dir}")

    def train(self):
        """主训练循环"""
        logger.info("开始训练...")
        for epoch in range(1, self.training_config.num_epochs + 1):
            self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            # 保存当前 Epoch 模型
            checkpoint_dir = f"checkpoint-epoch-{epoch}"
            self.save_checkpoint(checkpoint_dir, epoch, val_metrics)
            logger.info(f"已保存 Epoch {epoch} 模型至: {checkpoint_dir}")

            # 保存最佳模型（仅保存最佳模型权重）
            if val_metrics['explicit_hard_accuracy'] > self.best_accuracy:
                # 删除旧的最佳模型目录
                if self.best_model_dir is not None and os.path.exists(self.best_model_dir):
                    shutil.rmtree(self.best_model_dir)
                    logger.info(f"已删除旧的最佳模型目录: {self.best_model_dir}")
                
                self.best_accuracy = val_metrics['explicit_hard_accuracy']
                self.best_model_dir = os.path.join(self.output_dir, 'best_model')
                # 保存新的最佳模型
                self.save_checkpoint('best_model', epoch, val_metrics)
                logger.info(f"保存新的最佳模型，显式难例准确率: {self.best_accuracy:.4f}")

        logger.info("训练完成！")

    def train_epoch(self, epoch: int):
        """单个训练轮次 - 包含在线难例挖掘 (Online Hard Negative Mining)"""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}/{self.training_config.num_epochs}")

        for i, batch in progress_bar:
            # 将数据移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 1. 前向传播获取所有 Embedding
            # 注意: dataset 中返回的是一个 anchor, 一个 positive, 一个 sampled negative, 一个 explicit hard negative, 以及可能的其他 negatives
            anchor_emb, positive_emb, sampled_negative_emb, explicit_hard_negative_emb, other_negative_emb, other_negatives_counts = self.model(**batch)          
                         
            # B. Random Negative Loss (使用 dataset 采样的 random negative)
            loss_random = self.loss_fn(anchor_emb, positive_emb, sampled_negative_emb)

            # C. Explicit Hard Negative Loss (新增：专门针对生成的细节错误样本)
            loss_explicit_hard = self.loss_fn(anchor_emb, positive_emb, explicit_hard_negative_emb)

            # D. Other Negatives Loss (新增：利用 negs[1:] 的所有负样本)
            loss_other = torch.tensor(0.0, device=self.device)
            if other_negative_emb is not None and other_negatives_counts is not None and other_negative_emb.size(0) > 0:
                # 需要根据 counts 重复 anchor 和 positive
                anchor_repeated = torch.repeat_interleave(anchor_emb, other_negatives_counts, dim=0)
                positive_repeated = torch.repeat_interleave(positive_emb, other_negatives_counts, dim=0)
                loss_other = self.loss_fn(anchor_repeated, positive_repeated, other_negative_emb)
            
            # 组合 Loss（使用配置的权重）
            # 这里我稍微调整了权重策略：
            # 0.5 * Random (保持基础区分度)
            # 1.0 * Explicit_Hard (新增：重点强调区分细节错误)
            # 1.0 * Other_Negs (新增：充分利用其余负样本)
            loss = self.random_negative_weight * loss_random + self.explicit_negative_weight * loss_explicit_hard + self.other_negative_weight * loss_other
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # 日志记录
            if (i + 1) % self.training_config.logging_steps == 0:
                avg_loss = total_loss / (i + 1)
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.4f}", 
                    l_other=f"{loss_other.item():.4f}",
                    l_rand=f"{loss_random.item():.4f}",
                    l_exp=f"{loss_explicit_hard.item():.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}"
                )
                
                # SwanLab 日志记录
                if self.swanlab_config.enable:
                    swanlab.log({
                        "train/loss": avg_loss,
                        "train/loss_random": loss_random.item(),
                        "train/loss_explicit_hard": loss_explicit_hard.item(),
                        "train/loss_other": loss_other.item(),
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch
                    })
        
        avg_train_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {epoch} 训练损失: {avg_train_loss:.4f}")

    def validate_epoch(self, epoch: int) -> dict:
        """单个验证轮次"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        correct_explicit_hard_predictions = 0 # 新增：显式难例准确率
        correct_other_predictions = 0         # 新增：其他负例准确率
        
        total_samples = 0
        total_other_pairs = 0                 # 新增：其他负例对总数
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"验证 Epoch {epoch}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                anchor_emb, positive_emb, sampled_negative_emb, explicit_hard_negative_emb, other_negative_emb, other_negatives_counts = self.model(**batch)
                
                # 1. 计算 Loss (包含 Explicit Hard Loss)
                loss_random = self.loss_fn(anchor_emb, positive_emb, sampled_negative_emb)
                loss_explicit = self.loss_fn(anchor_emb, positive_emb, explicit_hard_negative_emb)

                loss_other = torch.tensor(0.0, device=self.device)
                if other_negative_emb is not None and other_negatives_counts is not None and other_negative_emb.size(0) > 0:
                     anchor_repeated = torch.repeat_interleave(anchor_emb, other_negatives_counts, dim=0)
                     positive_repeated = torch.repeat_interleave(positive_emb, other_negatives_counts, dim=0)
                     loss_other = self.loss_fn(anchor_repeated, positive_repeated, other_negative_emb)

                loss = loss_random + loss_explicit + loss_other # 验证集 Loss 简单求和即可
                total_loss += loss.item()
                
                # 2. 计算标准准确率 (Global Random Accuracy)
                # 对比: Anchor(图) vs Positive(正确文本) vs Sampled_Negative(全局随机文本)
                dist_pos = F.pairwise_distance(anchor_emb, positive_emb)
                dist_neg = F.pairwise_distance(anchor_emb, sampled_negative_emb)
                correct_predictions += torch.sum(dist_pos < dist_neg).item()

                # 3. 计算 Explicit Hard 准确率 (Explicit Hard Accuracy)
                # 对比: Anchor(图) vs Positive(正确文本) vs Explicit_Hard(negs[0])
                dist_explicit_hard_neg = F.pairwise_distance(anchor_emb, explicit_hard_negative_emb)
                correct_explicit_hard_predictions += torch.sum(dist_pos < dist_explicit_hard_neg).item()

                # 4. 计算 Other Negatives 准确率 (Local Random Accuracy)
                # 对比: Anchor(图) vs Positive(正确文本) vs Other_Negs(negs[1:])
                if other_negative_emb is not None and other_negatives_counts is not None and other_negative_emb.size(0) > 0:
                    # 重复 dist_pos 以匹配 other_negatives 的数量
                    dist_pos_repeated = torch.repeat_interleave(dist_pos, other_negatives_counts, dim=0)
                    
                    # 计算 anchor 与所有 other negatives 的距离
                    # 我们需要将 anchor 重复以匹配 flattened 的 other negatives
                    anchor_repeated_for_dist = torch.repeat_interleave(anchor_emb, other_negatives_counts, dim=0)
                    dist_other_neg = F.pairwise_distance(anchor_repeated_for_dist, other_negative_emb)
                    
                    correct_other_predictions += torch.sum(dist_pos_repeated < dist_other_neg).item()
                    total_other_pairs += other_negative_emb.size(0)
                
                total_samples += anchor_emb.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        explicit_hard_accuracy = correct_explicit_hard_predictions / total_samples
        other_accuracy = correct_other_predictions / total_other_pairs if total_other_pairs > 0 else 0.0
        
        logger.info(f"验证指标: loss={avg_loss:.4f}, global_acc={accuracy:.4f}, explicit_hard_acc={explicit_hard_accuracy:.4f}, other_acc={other_accuracy:.4f}")
        
        # SwanLab 验证集日志记录
        if self.swanlab_config.enable:
            swanlab.log({
                "val/loss": avg_loss,
                "val/global_accuracy": accuracy,
                "val/explicit_hard_accuracy": explicit_hard_accuracy,
                "val/other_accuracy": other_accuracy,
                "val/epoch": epoch
            })
            
        # 返回指标
        return {
            'loss': avg_loss, 
            'pairwise_accuracy': accuracy, # Global Accuracy
            'explicit_hard_accuracy': explicit_hard_accuracy,
            'other_accuracy': other_accuracy
        }

    def save_checkpoint(self, dirname: str, epoch: int, metrics: dict):
        """
        保存模型检查点。
        改为创建目录，并分别保存：
        1. 模型权重 (safetensors, 分片)
        2. 训练器状态 (optimizer, scheduler, epoch, etc. -> .pt)
        """
        save_dir = os.path.join(self.output_dir, dirname)
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 保存模型权重 (委托给 Model 类的 save_pretrained)
        self.model.save_pretrained(save_dir)
        
        # 2. 保存训练器状态 (依然使用 torch.save)
        trainer_state = {
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_accuracy': self.best_accuracy
        }
        torch.save(trainer_state, os.path.join(save_dir, "trainer_state.pt"))
        
        # 3. 保存 Processor 配置（如果存在）
        if self.processor is not None:
            self.processor.save_pretrained(save_dir)
            logger.info(f"Processor 配置已保存至: {save_dir}")

        logger.info(f"检查点已保存至: {save_dir}")
