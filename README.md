# Caltech-101 图像分类项目

本项目实现了使用预训练的卷积神经网络在Caltech-101数据集上进行图像分类的完整解决方案。项目包含数据预处理、模型微调、训练、测试和结果分析等完整流程。

## 项目特性

- 🔥 支持多种预训练模型：ResNet-18, ResNet-50, AlexNet, VGG-16, DenseNet-121
- 📊 多种实验设置：预训练微调、特征冻结、从头训练、不同学习率策略
- 📈 完整的训练监控：TensorBoard可视化、训练曲线、混淆矩阵
- 🎯 符合Caltech-101标准：每类30个训练样本，其余用于测试
- 📋 详细的实验对比：预训练vs从头训练的性能对比分析
- 🚀 易于使用：一键运行所有实验或单独运行特定实验

## 环境要求

- Python 3.7+
- PyTorch 1.13.0+
- CUDA (可选，用于GPU加速)

## 安装

1. 克隆项目：
```bash
git clone <your-repo-url>
cd caltech-101-classification
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备数据集：
```bash
# 确保 101_ObjectCategories 目录在项目根目录下
# 如果数据集在其他位置，可以通过 --data-dir 参数指定
```

## 快速开始

### 1. 数据集分析
```bash
python main.py --analyze-only
```

### 2. 运行所有实验（ResNet-18）
```bash
python main.py --models resnet18
```

### 3. 运行特定实验
```bash
python main.py --models resnet18 --experiments pretrained_finetune from_scratch
```

### 4. 多模型对比
```bash
python main.py --models resnet18 alexnet --experiments pretrained_finetune
```

### 5. 测试已训练的模型
```bash
python test.py --model models/resnet18_pretrained_finetune_best.pth
```

### 6. 预测单张图像
```bash
python test.py --model models/resnet18_pretrained_finetune_best.pth --image path/to/image.jpg
```

## 实验配置

项目预设了4种实验配置：

### 1. pretrained_finetune（推荐）
- 使用ImageNet预训练权重
- 微调所有层（backbone使用较小学习率）
- 学习率：0.001（分类器）/ 0.0001（backbone）
- 训练轮数：50

### 2. pretrained_transfer
- 使用ImageNet预训练权重
- 冻结特征提取层，只训练分类器
- 学习率：0.01
- 训练轮数：30

### 3. from_scratch
- 从随机初始化开始训练
- 所有层都训练
- 学习率：0.01
- 训练轮数：100

### 4. low_lr_finetune
- 使用ImageNet预训练权重
- 使用更小的学习率进行微调
- 学习率：0.0001
- 使用余弦学习率调度
- 训练轮数：60

## 命令行参数

### main.py
```bash
python main.py [OPTIONS]

选项:
  --models MODELS [MODELS ...]     要测试的模型 (resnet18, resnet50, alexnet, vgg16, densenet121)
  --experiments EXPERIMENTS [...]  要运行的实验 (pretrained_finetune, pretrained_transfer, from_scratch, low_lr_finetune)
  --data-dir DATA_DIR             数据集目录路径 (默认: 101_ObjectCategories)
  --batch-size BATCH_SIZE         批大小 (默认: 32)
  --num-workers NUM_WORKERS       数据加载进程数 (默认: 4)
  --analyze-only                  仅分析数据集，不进行训练
```

### test.py
```bash
python test.py [OPTIONS]

选项:
  --model MODEL                   单个模型文件路径
  --models MODELS [MODELS ...]    多个模型文件路径
  --image IMAGE                   单张图像路径（用于预测）
  --data-dir DATA_DIR            数据集目录路径
  --batch-size BATCH_SIZE        批大小
  --output-dir OUTPUT_DIR        输出目录
  --device {auto,cpu,cuda}       计算设备
```

## 输出文件说明

训练完成后，会在以下目录生成结果：

### results/
- `comparison_results.csv` - 所有实验结果对比表
- `accuracy_comparison.png` - 准确率对比图
- `pretraining_comparison.png` - 预训练vs从头训练对比
- `training_curves_comparison.png` - 训练曲线对比
- `{model}_{experiment}_results.json` - 单个实验的详细结果
- `{model}_{experiment}_confusion_matrix.png` - 混淆矩阵
- `{model}_{experiment}_training_curves.png` - 训练曲线

### models/
- `{model}_{experiment}_best.pth` - 最佳模型权重

### runs/
- TensorBoard日志文件

## TensorBoard可视化

启动TensorBoard查看训练过程：
```bash
tensorboard --logdir runs
```

然后在浏览器中访问 `http://localhost:6006`

## 实验结果示例

典型的实验结果（ResNet-18）：

| 实验类型 | 测试准确率 | 训练时间 | 说明 |
|---------|-----------|---------|------|
| pretrained_finetune | ~85-90% | ~30分钟 | 最佳性能 |
| pretrained_transfer | ~75-80% | ~15分钟 | 快速训练 |
| from_scratch | ~60-70% | ~60分钟 | 基线对比 |
| low_lr_finetune | ~83-88% | ~45分钟 | 稳定训练 |

## 项目结构

```
caltech-101-classification/
├── main.py                 # 主训练脚本
├── test.py                # 测试脚本
├── config.py              # 配置文件
├── dataset.py             # 数据集处理
├── models.py              # 模型定义
├── trainer.py             # 训练器
├── requirements.txt       # 依赖列表
├── README.md             # 项目说明
├── 101_ObjectCategories/ # 数据集目录
├── results/              # 实验结果
├── models/               # 训练好的模型
└── runs/                 # TensorBoard日志
```

## 自定义实验

可以通过修改 `config.py` 中的 `EXPERIMENTS` 字典来添加自定义实验配置：

```python
EXPERIMENTS = {
    'my_experiment': {
        'pretrained': True,
        'freeze_features': False,
        'learning_rate': 0.005,
        'num_epochs': 40,
        'lr_scheduler': 'step',
        'step_size': 15,
        'gamma': 0.1
    }
}
```

## 性能优化建议

1. **GPU加速**：如果有GPU，确保安装CUDA版本的PyTorch
2. **批大小**：根据GPU内存调整批大小（通常32-64效果较好）
3. **数据加载**：适当增加num_workers数量（通常设为CPU核心数的一半）
4. **混合精度**：对于大模型可以考虑使用AMP加速训练

## 常见问题

### Q: 训练时显存不足怎么办？
A: 减小batch_size，或者使用梯度累积技术。

### Q: 如何使用自己的数据集？
A: 修改dataset.py中的Caltech101Dataset类，适配你的数据格式。

### Q: 如何添加新的模型？
A: 在models.py中的Caltech101Classifier类中添加新模型的支持。

### Q: 训练速度太慢怎么办？
A: 确保使用GPU，减少num_epochs，或者使用更小的模型。

## 引用

如果这个项目对你的研究有帮助，请引用：

```bibtex
@misc{caltech101-classification,
  title={Caltech-101 Image Classification with Pre-trained CNN},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-username/caltech-101-classification}}
}
```

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 联系方式

如有问题，请通过以下方式联系：
- GitHub Issues: [项目Issues页面]
- Email: [your.email@example.com] 