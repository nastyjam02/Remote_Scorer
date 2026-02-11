"""
主训练脚本
"""
import logging
import torch
from transformers import AutoProcessor, get_scheduler
from torch.optim import AdamW

from config import ModelConfig, DataConfig, TrainingConfig, SwanLabConfig
from model import ImageTextEncoder
from dataset import create_dataloader
from loss import create_loss_fn
from trainer import Trainer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    # 1. 加载配置
    model_config = ModelConfig()
    data_config = DataConfig()
    training_config = TrainingConfig()
    swanlab_config = SwanLabConfig()

    # 2. 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 3. 初始化模型和 processor
    if training_config.resume_from_checkpoint:
        logging.info(f"从检查点加载模型: {training_config.resume_from_checkpoint}")
        # 使用 from_pretrained 加载微调后的模型 (包含 LoRA 和 投影头)
        # 传入 model_config 作为回退配置，以防检查点中缺失 model_config.json
        model = ImageTextEncoder.from_pretrained(training_config.resume_from_checkpoint, config=model_config)
        # 加载 processor (假设检查点目录或原始模型路径中有)
        # 优先尝试从检查点加载，如果失败则回退到 config 中的 model_path
        try:
            processor = AutoProcessor.from_pretrained(training_config.resume_from_checkpoint, trust_remote_code=True)
            logging.info("从检查点加载 Processor 成功")
        except Exception:
            logging.warning("从检查点加载 Processor 失败，回退到原始模型路径")
            processor = AutoProcessor.from_pretrained(model_config.model_path, trust_remote_code=True)
    else:
        logging.info("从头初始化模型")
        model = ImageTextEncoder(model_config)
        processor = AutoProcessor.from_pretrained(model_config.model_path, trust_remote_code=True)
    
    model = model.to(device)

    # 4. 创建数据加载器
    logging.info("创建数据加载器...")
    train_loader = create_dataloader(
        file_path=data_config.train_file,
        config=data_config,
        batch_size=training_config.batch_size,
        processor=processor,
        is_train=True
    )
    val_loader = create_dataloader(
        file_path=data_config.val_file,
        config=data_config,
        batch_size=training_config.batch_size,
        processor=processor,
        is_train=False
    )
    logging.info(f"训练数据: {len(train_loader)} 批次")
    logging.info(f"验证数据: {len(val_loader)} 批次")

    # 5. 初始化损失函数
    loss_fn = create_loss_fn(training_config.margin)

    # 6. 初始化优化器和学习率调度器
    optimizer = AdamW(
        model.get_trainable_parameters(), 
        lr=training_config.learning_rate, 
        weight_decay=training_config.weight_decay
    )
    
    num_training_steps = training_config.num_epochs * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=training_config.num_warmup_steps,
        num_training_steps=num_training_steps
    )

    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        swanlab_config=swanlab_config,
        processor=processor
    )
    
    try:
        trainer.train()
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}", exc_info=True)
    finally:
        logging.info("训练完成！")

if __name__ == "__main__":
    main()
