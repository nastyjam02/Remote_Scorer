"""
推理模块
"""
import torch
import os
import logging
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import ImageTextEncoder, ModelConfig
from config import DataConfig
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

class ImageTextDataset(Dataset):
    """
    用于批量推理的数据集类。
    负责在 CPU 上预读取和处理图片，减轻 GPU 等待时间。
    """
    def __init__(self, image_paths: list, texts: list, image_size: int = 512):
        """
        Args:
            image_paths: 图片路径列表
            texts: 对应的文本列表
            image_size: 图片 Resize 的大小。强制统一尺寸，否则无法进行 Batch 推理。
        """
        assert len(image_paths) == len(texts), "图片和文本列表长度必须一致"
        self.image_paths = image_paths
        self.texts = texts
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        text = self.texts[idx]
        
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                # 强制 Resize 以确保 Batch 处理时每张图产生的 Token 数量一致
                if self.image_size:
                    image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
            else:
                logger.warning(f"图像文件不存在: {image_path}")
                # 返回全黑图作为占位符
                image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
                
        except Exception as e:
            logger.error(f"读取图像失败 {image_path}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
            
        return image, text

def collate_fn(batch):
    """
    自定义 collate_fn， batch 中包含 PIL.Image 对象，默认的 collate 无法处理
    """
    images, texts = zip(*batch)
    return list(images), list(texts)

class SimilarityScorer:
    """
    使用模型计算图像和文本的相似度分数。
    """
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"正在加载模型检查点: {checkpoint_path}")
        
        # 使用 from_pretrained 直接加载模型
        if os.path.isdir(checkpoint_path):
             self.model = ImageTextEncoder.from_pretrained(checkpoint_path).to(self.device)
        else:
             raise ValueError(f"checkpoint_path 必须是一个包含模型文件的目录: {checkpoint_path}")

        self.model.eval()
        self.model_config = self.model.config
        
        logger.info(f"加载 Processor: {self.model_config.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_config.model_path, trust_remote_code=True)
        
        logger.info("相似度评分器初始化完成")

    def score(self, image_path: str, text: str) -> float:
        """
         对单个图文对进行评分
        """
        if not os.path.exists(image_path):
            logger.error(f"图像文件不存在: {image_path}")
            return 0.0
            
        # 复用 score_batch 的逻辑，但要注意单张图片的处理
        try:
            image = Image.open(image_path).convert("RGB")
            # 保持一致性
            image = image.resize((512, 512), Image.BICUBIC)
            
            scores = self.score_batch([image], [text])
            return scores[0]
        except Exception as e:
            logger.error(f"评分失败: {e}", exc_info=True)
            return 0.0

    def score_batch(self, images: list, texts: list) -> list:
        """
        批量推理
        Args:
            images: PIL.Image 对象列表
            texts: 文本字符串列表
        Returns:
            list[float]: 分数列表
        """
        if not images:
            return []
            
        try:
            #批量处理图像
            image_inputs = self.processor.image_processor(
                images=images,
                return_tensors="pt"
            )
            pixel_values = image_inputs['pixel_values'].to(self.device)
            image_grid_thw = image_inputs['image_grid_thw'].to(self.device)
            
            # 批量处理文本
            text_inputs = self.processor(
                text=texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # 推理
            with torch.no_grad(): 
                image_embedding = self.model.encode_image(pixel_values, image_grid_thw)
                text_embedding = self.model.encode_text(text_inputs.input_ids, text_inputs.attention_mask)

                similarity = F.cosine_similarity(image_embedding, text_embedding, dim=1)
                
                # 归一化
                scores = (similarity + 1) / 2
                
            return scores.cpu().tolist()
            
        except Exception as e:
            logger.error(f"批量评分发生错误: {e}", exc_info=True)
            # 发生错误时返回全 0
            return [0.0] * len(images)

    def predict_dataset(self, image_paths: list, texts: list, batch_size: int = 32, num_workers: int = 4, image_size: int = 512) -> list:
        """
        使用 DataLoader 进行高性能批量推理
        """
        logger.info(f"开始批量推理: 总样本数 {len(image_paths)}, Batch Size {batch_size}, Workers {num_workers}")
        
        dataset = ImageTextDataset(image_paths, texts, image_size=image_size)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True # 加速 CPU -> GPU 传输
        )
        
        all_scores = []
        
        for batch_images, batch_texts in tqdm(dataloader, desc="Inference"):
            batch_scores = self.score_batch(batch_images, batch_texts)
            all_scores.extend(batch_scores)
            
        return all_scores

