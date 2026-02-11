"""
双塔图文编码器模型
"""
import os
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
from safetensors.torch import save_file, load_file
import logging
import json

from config import ModelConfig

logger = logging.getLogger(__name__)

class ProjectionHead(nn.Module):
    """
    将骨干网络输出的特征投影到共享的嵌入空间。
    支持动态层数构建。
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        
        # 如果 num_layers 小于 1，默认至少有一层
        if num_layers < 1:
            num_layers = 1
            
        layers = []
        # 输入维度到隐藏层维度的映射
        # 如果只有一层，直接 input -> output
        # 如果多层，input -> hidden -> ... -> hidden -> output
        
        # 为了保持与之前的逻辑一致，我们让 hidden_dim = output_dim
        hidden_dim = output_dim
        
        # 构建前面的层 (1 到 num_layers - 1)
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
            
        # 构建最后一层
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class AttentionPooling(nn.Module):
    """
    注意力池化层：通过学习每个token的权重来进行加权平均。
    比平均池化更能捕捉细粒度语义信息。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        # 初始化为 0，使得初始状态下的 Attention Score 全为 0
        # 经过 Softmax 后，所有权值相等 (1/N)
        # 此时 Attention Pooling 等价于 Mean Pooling
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: [batch_size, seq_len, hidden_dim]
        # attention_mask: [batch_size, seq_len]
        
        # 1. 计算每个 token 的分数
        # scores: [batch_size, seq_len, 1]
        scores = self.attention(last_hidden_state)
        
        # 2. 处理 Padding，将 Padding 位置的分数设为极小值 (-inf)
        # mask: 1 for valid, 0 for padding
        # (1.0 - attention_mask) * -1e9 会让 padding 位置变成很大的负数
        mask_value = -1e9
        scores = scores + (1.0 - attention_mask.unsqueeze(-1)) * mask_value
        
        # 3. 计算权重 (Softmax)
        # weights: [batch_size, seq_len, 1]
        weights = torch.softmax(scores, dim=1)
        
        # 4. 加权求和
        # [batch_size, seq_len, 1] * [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, hidden_dim]
        # sum(dim=1) -> [batch_size, hidden_dim]
        embeddings = torch.sum(weights * last_hidden_state, dim=1)
        
        return embeddings

class ImageTextEncoder(nn.Module):
    """
    双塔模型，用于分别编码图像和文本到同一语义空间。
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        logger.info(f"加载预训练模型: {config.model_path}")
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_path, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )
        
        self.vision_encoder = self.backbone.visual
        self.language_model = self.backbone.model.language_model

        # 获取骨干网络的输出维度
        # Vision Tower 和 Text Tower 的输出维度实际上都是 LLM 的 hidden size
        hidden_dim = self.backbone.config.hidden_size

        # 创建独立的图像和文本投影头
        # 使用 config 中的 num_projection_layers，如果没有则默认为 2 (保持兼容性)
        num_layers = getattr(config, 'num_projection_layers', 2)
        self.image_projection = ProjectionHead(hidden_dim, config.projection_dim, config.dropout, num_layers=num_layers)
        self.text_projection = ProjectionHead(hidden_dim, config.projection_dim, config.dropout, num_layers=num_layers)
        self.image_attention_pool = AttentionPooling(hidden_dim)
        self.text_attention_pool = AttentionPooling(hidden_dim)

        self._freeze_backbone()

        # 配置 LoRA
        if getattr(config, 'use_lora', False):
            logger.info("正在为视觉编码器配置 LoRA...")
            # Qwen2.5-VL Vision Encoder 使用 'qkv' 作为注意力层
            
            # 1. 处理 target_modules (防御性编程)
            target_modules = config.lora_target_modules
            
            # 处理 "None" 字符串、["None"] 列表或 None 对象的情况
            if target_modules is None or target_modules == "None" or target_modules == ["None"]:
                target_modules = ["qkv"]
            
            # 再次确认如果是空列表，也使用默认值
            if not target_modules:
                target_modules = ["qkv"]
                
            logger.info(f"LoRA Target Modules set to: {target_modules}")
            
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                modules_to_save=[],
            )
            self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)
            self.vision_encoder.print_trainable_parameters()
            logger.info("LoRA 配置完成")
        
        # 确保投影头的数据类型与骨干网络一致
        self.image_projection.to(dtype=self.backbone.dtype)
        self.text_projection.to(dtype=self.backbone.dtype)
        self.image_attention_pool.to(dtype=self.backbone.dtype)
        self.text_attention_pool.to(dtype=self.backbone.dtype)
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # ln(1/0.07) ≈ 2.6592

        logger.info("双塔模型初始化完成")

    def _freeze_backbone(self):
        """根据配置冻结骨干网络的参数"""
        if self.config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            logger.info("视觉编码器已冻结")
        
        if self.config.freeze_text_encoder:
            for param in self.language_model.parameters():
                param.requires_grad = False
            logger.info("文本编码器已冻结")
            
    def get_trainable_parameters(self):
        """获取可训练的参数"""
        params = list(self.image_projection.parameters()) + list(self.text_projection.parameters())
        params += list(self.image_attention_pool.parameters())
        params += list(self.text_attention_pool.parameters())
        
        # 如果启用了 LoRA，添加 Vision Encoder 的可训练参数
        if hasattr(self, 'vision_encoder') and any(p.requires_grad for p in self.vision_encoder.parameters()):
            params += [p for p in self.vision_encoder.parameters() if p.requires_grad]
            
        return params

    def encode_image(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        # [total_tokens, hidden_size] (因为是动态分辨率，所有batch的token被压扁了)
        image_features = self.vision_encoder(pixel_values, grid_thw=grid_thw) 
        
        # 恢复 Batch 结构
        # 由于输入图像大小固定 (256x256)，每张图的 token 数是相同的
        # 我们可以直接 reshape
        batch_size = grid_thw.shape[0]
        hidden_size = image_features.shape[-1]
        
        # [batch_size, tokens_per_image, hidden_size]
        image_features = image_features.view(batch_size, -1, hidden_size)
        
        # --- 改用注意力池化 (Attention Pooling) ---
        # 构造全 1 的 attention mask (假设所有 token 有效，因为输入尺寸固定)
        image_attention_mask = torch.ones(image_features.shape[:2], device=image_features.device)
        pooled_features = self.image_attention_pool(image_features, image_attention_mask)
        
        # 确保类型匹配
        first_layer = next(self.image_projection.model.modules())
        if isinstance(first_layer, nn.Sequential):
             first_layer = self.image_projection.model[0]
        pooled_features = pooled_features.to(dtype=first_layer.weight.dtype)
        
        # --- 原平均池化 (Mean Pooling) [已弃用] ---
        # pooled_features = image_features.mean(dim=1)
        # ----------------------------------------
        
        # 通过投影头
        image_embedding = self.image_projection(pooled_features)
        
        # L2 归一化
        image_embedding = nn.functional.normalize(image_embedding, p=2, dim=1)
        
        return image_embedding

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """编码文本"""
        # [batch_size, seq_len, hidden_size]
        text_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        last_hidden_state = text_outputs.last_hidden_state
        
        # --- 改用注意力池化 (Attention Pooling) ---
        # 它可以自动学习每个 token 的权重，从而关注到更具区分性的词 (如 "dark" vs "light")
        pooled_features = self.text_attention_pool(last_hidden_state, attention_mask)

        # --- 原平均池化 (Mean Pooling) [已弃用] ---
        # 使用 attention mask 排除 padding tokens
        # mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        # sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        # pooled_features = sum_embeddings / sum_mask
        # ----------------------------------------

        # 将特征转回投影头的数据类型 (因为上面的计算使用了 float32)
        # 获取投影头第一层的权重类型
        first_layer = next(self.text_projection.model.modules())
        # skip sequential container
        if isinstance(first_layer, nn.Sequential):
             first_layer = next(first_layer.modules())
             # skip sequential container itself again if needed or just access via index
             first_layer = self.text_projection.model[0]
        
        pooled_features = pooled_features.to(dtype=first_layer.weight.dtype)

        # 通过投影头
        text_embedding = self.text_projection(pooled_features)
        
        # L2 归一化
        text_embedding = nn.functional.normalize(text_embedding, p=2, dim=1)
        
        return text_embedding

    def forward(self, 
                anchor_pixel_values: torch.Tensor,
                anchor_image_grid_thw: torch.Tensor,
                positive_input_ids: torch.Tensor,
                positive_attention_mask: torch.Tensor,
                negative_input_ids: torch.Tensor,
                negative_attention_mask: torch.Tensor,
                hard_negative_input_ids: torch.Tensor = None,
                hard_negative_attention_mask: torch.Tensor = None,
                other_negative_input_ids: torch.Tensor = None,
                other_negative_attention_mask: torch.Tensor = None,
                other_negatives_counts: torch.Tensor = None):
        """
        用于训练的前向传播，接收一个三元组（或四元组/五元组，如果包含显式难例和其他负例）。
        """
        anchor_embedding = self.encode_image(anchor_pixel_values, anchor_image_grid_thw)
        positive_embedding = self.encode_text(positive_input_ids, positive_attention_mask)
        negative_embedding = self.encode_text(negative_input_ids, negative_attention_mask)
        
        explicit_hard_negative_embedding = None
        if hard_negative_input_ids is not None and hard_negative_attention_mask is not None:
            explicit_hard_negative_embedding = self.encode_text(hard_negative_input_ids, hard_negative_attention_mask)
            
        other_negative_embeddings = None
        if other_negative_input_ids is not None and other_negative_attention_mask is not None:
            other_negative_embeddings = self.encode_text(other_negative_input_ids, other_negative_attention_mask)
            
        return anchor_embedding, positive_embedding, negative_embedding, explicit_hard_negative_embedding, other_negative_embeddings, other_negatives_counts

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        """
        保存模型到指定目录，支持 safetensors 和分片保存。
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. 保存骨干网络
        # 优化：如果使用了 LoRA 且骨干网络被冻结，我们不需要保存庞大的基础权重
        # 只需要保存 LoRA 权重和投影头，加载时会复用原始底座
        
        save_backbone = True
        if getattr(self.config, 'use_lora', False):
             # 检查是否所有骨干参数都冻结了
             if self.config.freeze_vision_encoder and self.config.freeze_text_encoder:
                 save_backbone = False
                 logger.info("检测到 LoRA 模式且骨干网络冻结，将跳过基础模型权重的保存以节省空间。")
        
        if save_backbone:
            logger.info(f"保存骨干网络权重到: {save_directory}")
            self.backbone.save_pretrained(
                save_directory, 
                safe_serialization=safe_serialization, 
                max_shard_size="2GB" 
            )
        else:
            # 即使不保存权重，我们最好也保存一下 tokenizer/processor 的配置
            # 这样用户可以直接指向这个目录加载 processor
            try:
                # 尝试保存 processor/tokenizer 配置（如果有的话）
                # 这里我们没有直接持有 processor，但可以通过 backbone.config 保存部分信息
                # 或者更简单，我们假设用户推理时会回溯到原始 model_path 加载 processor
                pass 
            except Exception as e:
                logger.warning(f"保存配置时出错: {e}")
        
        # 如果使用了 LoRA，保存 adapter
        if getattr(self.config, 'use_lora', False) and hasattr(self, 'vision_encoder'):
            # 检查 vision_encoder 是否是 PeftModel
            from peft import PeftModel
            if isinstance(self.vision_encoder, PeftModel):
                logger.info(f"保存 LoRA 权重到: {save_directory}")
                
                # 修复 PEFT 警告: Could not find a config file
                # 手动设置 base_model_name_or_path，让 PEFT 知道底座在哪里
                if not hasattr(self.vision_encoder.peft_config['default'], 'base_model_name_or_path'):
                     self.vision_encoder.peft_config['default'].base_model_name_or_path = self.config.model_path
                
                self.vision_encoder.save_pretrained(save_directory, safe_serialization=safe_serialization)
        
        # 2. 保存投影头权重 (自定义保存)
        logger.info(f"保存投影头权重到: {save_directory}")
        # 收集所有投影头的权重
        heads_state_dict = {}
        for name, param in self.image_projection.named_parameters():
            heads_state_dict[f"image_projection.{name}"] = param
        for name, param in self.text_projection.named_parameters():
            heads_state_dict[f"text_projection.{name}"] = param
        for name, param in self.image_attention_pool.named_parameters():
            heads_state_dict[f"image_attention_pool.{name}"] = param
        for name, param in self.text_attention_pool.named_parameters():
            heads_state_dict[f"text_attention_pool.{name}"] = param
        # 还有可学习的 logit_scale
        heads_state_dict["logit_scale"] = self.logit_scale
            
        if safe_serialization:
            save_file(heads_state_dict, os.path.join(save_directory, "heads.safetensors"))
        else:
            torch.save(heads_state_dict, os.path.join(save_directory, "heads.bin"))
            
        # 3. 保存 ModelConfig
        config_dict = self.config.__dict__
        # 过滤掉不可序列化的对象（如果有）
        config_path = os.path.join(save_directory, "model_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
        logger.info("模型保存完成")

    @classmethod
    def from_pretrained(cls, load_directory: str, config=None):
        """
        从指定目录加载模型。
        """
        logger.info(f"正在从 {load_directory} 加载模型...")
        
        # 1. 加载配置
        config_path = os.path.join(load_directory, "model_config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            
            # 重建 config 对象
            class ConfigObject:
                pass
            loaded_config = ConfigObject()
            for k, v in config_dict.items():
                setattr(loaded_config, k, v)
            config = loaded_config
            
        except FileNotFoundError:
            if config is None:
                raise FileNotFoundError(f"未找到配置文件 {config_path}，且未提供回退配置。")
            logger.warning(f"未找到配置文件 {config_path}，使用提供的回退配置。")
            # config 已经由参数提供，直接使用
        
        # 2. 初始化模型结构
        # 关键修改：不要将 config.model_path 修改为 load_directory
        # 我们希望 backbone 从原始的预训练模型路径加载（基础底座）
        # 然后再加载 LoRA 权重
        
        model = cls(config)
        
        # 3. 如果使用了 LoRA，显式加载 LoRA 权重
        if getattr(config, 'use_lora', False):
            from peft.utils import set_peft_model_state_dict
            
            adapter_path = os.path.join(load_directory, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                logger.info(f"发现 LoRA 权重，正在加载: {adapter_path}")
                adapters_weights = load_file(adapter_path)
                # 加载到 vision_encoder
                set_peft_model_state_dict(model.vision_encoder, adapters_weights)
            else:
                logger.warning(f"配置显示使用了 LoRA，但在 {load_directory} 未找到 adapter_model.safetensors")

        # 4. 加载投影头权重
        heads_path = os.path.join(load_directory, "heads.safetensors")
        if os.path.exists(heads_path):
            heads_state_dict = load_file(heads_path)
        else:
            heads_path = os.path.join(load_directory, "heads.bin")
            if os.path.exists(heads_path):
                heads_state_dict = torch.load(heads_path, map_location="cpu")
            else:
                logger.warning(f"未找到投影头权重文件 heads.safetensors 或 heads.bin")
                heads_state_dict = {}
            
        # 加载权重
        if heads_state_dict:
            missing_keys, unexpected_keys = model.load_state_dict(heads_state_dict, strict=False)
            if unexpected_keys:
                logger.warning(f"加载投影头时发现未预期的键: {unexpected_keys}")
        
        logger.info("模型加载完成")
        return model
