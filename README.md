# 遥感图像-文本匹配模型

本项目基于 Qwen2.5-VL 预训练模型，采用双塔（Dual-Encoder）架构进行微调，旨在学习遥感图像与其文本描述之间的语义匹配关系。

## 1. 项目架构

与之前版本不同，本项目采用双塔模型结构，其核心思想是为图像和文本分别训练独立的编码器，将它们映射到同一个共享的语义空间中。

- **图像塔 (Image Tower)**: 使用 Qwen-VL 的视觉编码器提取图像特征，并通过一个投影头（Projection Head）生成最终的图像嵌入向量。
- **文本塔 (Text Tower)**: 使用 Qwen-VL 的语言模型提取文本特征，并通过一个结构相同的投影头生成文本嵌入向量。
- **训练目标**: 采用三元组损失函数 (Triplet Margin Loss) 作为优化目标。在训练过程中，模型会学习拉近匹配的图文对在嵌入空间中的距离，同时推远不匹配的图文对。
- **评分机制**: 图文匹配的最终得分通过计算图像嵌入和文本嵌入之间的**余弦相似度**来获得。

这个架构从根本上解决了旧模型将图文特征“平均化”处理的缺陷，使模型能够真正学习到跨模态的语义对齐能力。

## 2. 环境准备

建议使用 Conda 创建独立的虚拟环境。

```bash
# 创建 conda 环境
conda create -n qwen-vl-match python=3.10
conda activate qwen-vl-match

# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate sentencepiece Pillow tqdm tensorboard

# 如果需要从源码安装最新版 transformers
# pip install git+https://github.com/huggingface/transformers.git
```

## 3. 数据准备

训练和验证数据需要遵循三元组格式的 JSON 文件。每个 JSON 对象包含以下字段：

- `image_name`: 图像文件名 (例如: `"123.tif"`)
- `positive`: 与图像匹配的正面文本描述 (例如: `"一片密集的城市建筑区"`)
- `negative`: 与图像不匹配的负面文本描述 (例如: `"大面积的农田和几条河流"`)

**文件结构示例:**

```
.
├── 遥感图像评分器/
│   ├── train.py
│   ├── model.py
│   └── ...
├── imgs/
│   ├── 1.tif
│   └── 2.tif
├── train_triplet_data.json
└── test_triplet_data.json
```

请确保 `config.py` 文件中的 `train_file`, `val_file` 和 `image_root` 指向正确的文件和目录。

## 4. 模型训练

直接运行主训练脚本即可开始训练。所有超参数都可以在 `config.py` 中进行调整。

```bash
python train.py
```

训练过程中的日志、TensorBoard 记录以及保存的模型检查点会自动存储在 `output/training_YYYYMMDD_HHMMSS/` 目录下。

## 5. 模型推理

使用 `inference_script.py` 对新的图文对进行评分。脚本支持两种模式：

### 单张图文对评分

```bash
python inference_script.py \
    --checkpoint "output/training_20251226_114445/best_model/"\
    --image_path "../imgs/168.tif" \
    --text "一片被建筑物覆盖的区域，中间有几条道路"
```

### 批量评分 (使用JSON文件)

**输入JSON文件 (`input.json`):**
```json
[
    {
        "image_path": "../imgs/168.tif",
        "text": "一片密集的居民区"
    },
    {
        "image_path": "../imgs/100.tif",
        "text": "大面积的裸露土地和荒漠"
    }
]
```

**运行命令:**
```bash
python inference_script.py \
    --checkpoint "output/training_20251226_143347/best_model" \
    --json_input "input.json" \
    --json_output "output_scores.json"
```

评分结果将保存在 `output_scores.json` 中。
