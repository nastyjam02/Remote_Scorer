这是一个持续进行的项目，我们在持续进行后续的图像翻译工作的研究。

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

## Licensing Information

本项目所使用的图片数据来源于 **UC Merced Land Use Dataset (UCM)**和**RSGPT: A Remote Sensing Vision Language Model and Benchmark**。对于UCM数据集我们通过半人工的方式进行caption标注，相比于原origin的label更加精细完整，RSGPT则直接使用提供的标注信息。
- 该数据集由加州大学默塞德分校（University of California, Merced）提供。
- **使用限制**：数据集中的所有图片及其相关标注仅可用于**学术研究用途**，严禁任何形式的商业用途。
- **引用说明**：如果您在研究中使用了本项目的相关内容，请务必引用 UCM 数据集的原始论文：

```bibtex
@inproceedings{Yang2010UCM,
  title={Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification},
  author={Yang, Yi and Newsam, Shawn},
  booktitle={ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS)},
  year={2010}
}
```

## 致谢

本项目在开发过程中深受以下开源项目和研究工作的启发与支持，特此致谢：

- **Qwen2.5-VL**: 感谢阿里云 Qwen 团队提供的卓越视觉语言模型 Qwen2.5-VL。本项目基于其强大的预训练能力构建了双塔编码器。
  - *Reference*: Bai, Shuai, et al. "Qwen2.5-VL Technical Report." arXiv preprint arXiv:2502.13923 (2025).

- **RSGPT / InstructBLIP**: 本项目的模型架构设计部分参考了 InstructBLIP 的思想，同时 RSGPT 在遥感领域的应用为我们提供了宝贵的思路。如果您之前不了解这些优秀的工作，强烈推荐您进行深入了解！
```bibtex
@article{hu2025rsgpt,
  title={Rsgpt: A remote sensing vision language model and benchmark},
  author={Hu, Yuan and Yuan, Jianlong and Wen, Congcong and Lu, Xiaonan and Liu, Yu and Li, Xiang},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={224},
  pages={272--286},
  year={2025},
  publisher={Elsevier}
}
```
