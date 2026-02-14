"""
数据集和数据加载器
"""
import os
import json
import logging
import random
from PIL import Image
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoProcessor

from config import DataConfig

logger = logging.getLogger(__name__)

class TripletDataset(Dataset):
    """
    用于加载三元组数据 (anchor, positive, negative) 的数据集。
    - anchor: 图像
    - positive: 匹配的文本描述
    - negative: 不匹配的文本描述
    """
    def __init__(self, data: List[Dict], config: DataConfig, processor: AutoProcessor, is_train: bool):
        self.data = data
        self.config = config
        self.processor = processor
        self.is_train = is_train
        
        # 定义图像变换
        if is_train and config.use_augmentation:
            # 训练集增强：Resize -> 随机翻转 -> 随机旋转 -> 颜色抖动
            logging.info("数据增强已开启...")
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5), # 遥感图像没有上下之分，垂直翻转是安全的
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ])
        else:
            # 验证集/测试集：仅 Resize
            self.transform = transforms.Resize((config.image_size, config.image_size), interpolation=transforms.InterpolationMode.BICUBIC)

        logger.info(f"创建了 {len(self.data)} 个三元组的数据集 ({'训练' if is_train else '验证'})")
        
        # 构建全局 Positive 文本池，用于 Global Random Negative 采样
        self.all_pos_texts = [d['pos'] for d in self.data if 'pos' in d and d['pos']]

        # 确定当前数据集使用的图像根目录
        if is_train:
            self.image_root = getattr(config, 'train_image_root', getattr(config, 'image_root', ''))
        else:
            self.image_root = getattr(config, 'val_image_root', getattr(config, 'image_root', ''))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str, str, str]:
        item = self.data[idx]
        
        #  加载图像 (Anchor) - 只加载，不转换 Tensor
        image_name = item['image']
        if image_name.startswith("imgs/"):
            image_name = os.path.basename(image_name)
            
        image_path = os.path.join(self.image_root, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"无法加载图像 {image_path}，程序将终止。错误: {e}")
            raise RuntimeError(f"图像加载失败: {image_path}") from e

        # 应用图像变换 (Resize + Augmentation)
        if hasattr(self, 'transform'):
            image = self.transform(image)

        #  处理文本 (Positive 和 Negative)
        positive_text = item['pos']
        
        # 获取显式 Hard Negative (negs[0])
        hard_negative_text = ""
        if 'negs' in item and len(item['negs']) > 0:
            hard_negative_text = item['negs'][0]
            # 如果没有 negs，直接报错退出
            logger.error(f"样本 {idx} 缺少 negs 字段，程序终止。")
            raise RuntimeError(f"数据格式错误：样本 {idx} 缺少 negs 字段")

        # 获取 Global Random Negative (用于语义区分能力学习)
        # 逻辑：从全局所有文本中随机选一个，确保不等于当前 positive_text
        while True:
            negative_text = random.choice(self.all_pos_texts)
            if negative_text != positive_text:
                break

        # 获取其他所有 Negative (negs[1:]), 用于额外的 Loss 计算
        other_negative_texts = []
        if 'negs' in item and len(item['negs']) > 1:
            other_negative_texts = item['negs'][1:]

        return image, positive_text, negative_text, hard_negative_text, other_negative_texts

def collate_fn(batch: List[Tuple], processor: AutoProcessor):
    """
    自定义的 collate_fn，用于将批次数据打包成模型所需的格式。
    使用 processor 来统一处理图像和文本。
    """
    anchor_images, positive_texts, negative_texts, hard_negative_texts, other_negative_texts_list = zip(*batch)
    
    #  使用 processor 处理图像
    image_inputs = processor.image_processor(
        images=list(anchor_images),
        videos=None,
        return_tensors="pt"
    )

    # DEBUG: 打印一次 image_grid_thw 以验证分辨率处理情况
    if not hasattr(collate_fn, "has_logged_grid"):
        logger.info(f"Debug: image_grid_thw shape: {image_inputs['image_grid_thw'].shape}")
        logger.info(f"Debug: image_grid_thw sample (first item): {image_inputs['image_grid_thw'][0]}")
        collate_fn.has_logged_grid = True
    
    positive_inputs = processor(
        text=list(positive_texts), 
        padding='max_length', 
        truncation=True, 
        max_length=128,
        return_tensors="pt"
    )
    
    negative_inputs = processor(
        text=list(negative_texts), 
        padding='max_length', 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    )
    
    hard_negative_inputs = processor(
        text=list(hard_negative_texts),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # 处理 Other Negatives
    # 将所有样本的 other_negatives 展平为一个大的列表进行 tokenization
    flat_other_negatives = []
    other_negatives_counts = [] # 记录每个样本有多少个 other_negative
    for other_negs in other_negative_texts_list:
        flat_other_negatives.extend(other_negs)
        other_negatives_counts.append(len(other_negs))
    
    other_negative_input_ids = None
    other_negative_attention_mask = None
    
    if len(flat_other_negatives) > 0:
        other_negative_inputs = processor(
            text=flat_other_negatives,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        other_negative_input_ids = other_negative_inputs.input_ids
        other_negative_attention_mask = other_negative_inputs.attention_mask
    
    return {
        "anchor_pixel_values": image_inputs['pixel_values'],
        "anchor_image_grid_thw": image_inputs['image_grid_thw'],  # 关键参数
        "positive_input_ids": positive_inputs.input_ids,
        "positive_attention_mask": positive_inputs.attention_mask,
        "negative_input_ids": negative_inputs.input_ids,
        "negative_attention_mask": negative_inputs.attention_mask,
        "hard_negative_input_ids": hard_negative_inputs.input_ids,
        "hard_negative_attention_mask": hard_negative_inputs.attention_mask,
        "other_negative_input_ids": other_negative_input_ids,
        "other_negative_attention_mask": other_negative_attention_mask,
        "other_negatives_counts": torch.tensor(other_negatives_counts, dtype=torch.long)
    }

def create_dataloader(
    file_path: str, 
    config: DataConfig, 
    batch_size: int, 
    processor: AutoProcessor, 
    is_train: bool
) -> DataLoader:
    """
    创建数据加载器。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = TripletDataset(data, config, processor, is_train)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )
