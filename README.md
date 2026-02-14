# 遥感图像-文本匹配模型

本项目灵感来源于clip，siglip系列的对比学习的训练方式，同时感谢QwenVL系列伟大的开源工作，我们基于 Qwen2.5-VL-3B 小型预训练模型，采用双塔（Dual-Encoder）架构进行微调，旨在学习遥感图像与其文本描述之间的语义匹配关系,旨来在众多对同一幅image的captions中挑选质量最优秀的one，我们相信这将为遥感图像理解和检索提供更精准的解决方案，同时为我们的后续的SAR2OPT的图像翻译工作提供技术支持。

## 1. 项目架构

本项目核心思想是为图像和文本分别训练独立的编码器，将它们映射到同一个共享的语义空间中。

- **图像塔 (Image Tower)**: 使用 Qwen-VL 的视觉编码器提取图像特征，并通过一个投影头（Projection Head）生成最终的图像嵌入向量。
- **文本塔 (Text Tower)**: 使用 Qwen-VL 的语言模型提取文本特征，并通过一个结构相同的投影头生成文本嵌入向量。
- **训练目标**: 采用FocalTripletLoss作为优化目标。在训练过程中，模型会学习拉近匹配的图文对在嵌入空间中的距离，同时推远不匹配的图文对。
- **评分机制**: 图文匹配的最终得分通过计算图像嵌入和文本嵌入之间的**余弦相似度**来获得。

## 2. 环境准备

建议使用新建Conda 创建独立的虚拟环境。

```bash
# 创建 conda 环境
conda create -n remote_scorer python=3.10
conda activate remote_scorer

# 安装核心依赖
conda env create -f environment.yml
pov：本项目建议以下相关库保持一致即可，如遇冲突建议回退到environment文件一致版本即可
- torch 2.9.0
- transformers : 4.57.3
- peft : 0.18.0
- qwen-vl-utils : 0.0.14
- accelerate : 1.11.0
```

## 3. 数据准备

训练和验证数据需要遵循三元组格式的 JSON 文件。每个 JSON 对象包含以下字段：

- `image_name`: 图像文件名 
- `positive`: 与图像匹配的正面文本描述 
- `negative`: 与图像不匹配的负面文本描述 


请确保 `config.py` 文件中的 `train_file`, `val_file` 和 `image_root` 指向正确的文件和目录。

## 4. 模型训练

直接运行主训练脚本即可开始训练。所有超参数都可以在 `config.py` 中进行调整。

```bash
python train.py
```

训练过程中的日志、TensorBoard 记录以及保存的模型检查点会自动存储。

## 5. 模型推理

使用 `inference_script.py` 对新的图文对进行评分。脚本支持两种模式：

### 单张图文对评分
### 批量评分 (使用JSON文件)

