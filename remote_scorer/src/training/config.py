"""
配置文件
"""
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """模型配置"""
    # 预训练模型路径
    model_path: str = "/data/oceanus_ctr/j-chenrui5-jk/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct"
    # 是否冻结视觉编码器
    freeze_vision_encoder: bool = True
    # 是否冻结文本编码器
    freeze_text_encoder: bool = True
    # 图像和文本投影头的输出维度
    projection_dim: int = 768
    # 投影头中的Dropout率
    dropout: float = 0
    # 投影头的层数 (Linear层数量)
    num_projection_layers: int = 2
    
    # LoRA 配置
    use_lora: bool = True
    lora_r: int = 16  # LoRA 的秩，越大参数越多但拟合能力越强，通常 8-64 之间
    lora_alpha: int = 32  # 缩放系数，通常设为 r 的 2 倍
    lora_dropout: float = 0.05
    # Qwen2.5-VL Vision Encoder 的 QKV 投影层名称为 "qkv"
    lora_target_modules: list = None  # 默认为 ["qkv"]

@dataclass
class DataConfig:
    """数据配置"""
    # 训练数据文件
    train_file: str = "UCM_train_triplet_data.json"
    # 验证数据文件
    val_file: str = "UCM_test_triplet_data.json"
    # 训练集图像根目录
    train_image_root: str = "/data/oceanus_ctr/j-chenrui5-jk/Score/imgs"
    # 验证集图像根目录
    val_image_root: str = "/data/oceanus_ctr/j-chenrui5-jk/Score/imgs"
    # 图像尺寸
    image_size: int = 512 
    # 是否启用数据增强
    use_augmentation: bool = False

@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练输出目录
    output_dir: str = "output"
    # 学习率
    learning_rate: float = 1e-4
    # 批次大小
    batch_size: int = 32
    # 训练轮次
    num_epochs: int = 30
    # 权重衰减
    weight_decay: float = 0.05
    # 三元组损失的边界 (margin)
    margin: float = 0.5
    # 学习率调度器的预热步数
    num_warmup_steps: int = 0
    # 日志记录步数
    logging_steps: int = 10
    # 每多少个epoch保存一次模型
    save_epochs: int = 1
    # 断点续训路径 (例如 "output/training_xxx/best_model")，为 None 则从头训练
    resume_from_checkpoint: str = None

@dataclass
class SwanLabConfig:
    """SwanLab 实验跟踪配置"""
    # 是否启用 SwanLab
    enable: bool = True
    # 项目名称
    project: str = "Remote-Sensing-Scorer"
    # 实验名称 (如果不填，SwanLab 会自动生成)
    experiment_name: str = None
    # 实验描述
    description: str = "Image-Text Alignment"
    # 运行模式: "cloud" (云端), "local" (本地), "disabled" (禁用)
    mode: str = "cloud"
