import os

class Config:
    # 数据集设置
    DATA_DIR = '101_ObjectCategories'
    NUM_CLASSES = 101  # Caltech-101有101个类别（不包括BACKGROUND_Google）
    
    # 数据预处理设置
    IMAGE_SIZE = 224  # ImageNet预训练模型的标准输入尺寸
    MEAN = [0.485, 0.456, 0.406]  # ImageNet数据集的均值
    STD = [0.229, 0.224, 0.225]   # ImageNet数据集的标准差
    
    # 训练设置
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # 模型设置
    PRETRAINED = True
    MODELS = ['resnet18', 'alexnet']  # 支持的模型
    
    # 实验设置
    EXPERIMENTS = {
        'pretrained_finetune': {
            'pretrained': True,
            'freeze_features': False,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'lr_scheduler': 'step',
            'step_size': 20,
            'gamma': 0.1
        },
        'pretrained_transfer': {
            'pretrained': True,
            'freeze_features': True,
            'learning_rate': 0.01,
            'num_epochs': 30,
            'lr_scheduler': 'step',
            'step_size': 15,
            'gamma': 0.1
        },
        'from_scratch': {
            'pretrained': False,
            'freeze_features': False,
            'learning_rate': 0.01,
            'num_epochs': 100,
            'lr_scheduler': 'step',
            'step_size': 30,
            'gamma': 0.1
        },
        'low_lr_finetune': {
            'pretrained': True,
            'freeze_features': False,
            'learning_rate': 0.0001,
            'num_epochs': 60,
            'lr_scheduler': 'cosine',
            'step_size': None,
            'gamma': None
        }
    }
    
    # 输出目录
    OUTPUT_DIR = 'results'
    LOG_DIR = 'logs'
    MODEL_DIR = 'models'
    
    # 设备设置
    DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 早停设置
    EARLY_STOPPING_PATIENCE = 10
    
    # 数据增强设置
    USE_AUGMENTATION = True
    
    # 验证集比例
    VAL_SPLIT = 0.2
    
    # TensorBoard设置
    TENSORBOARD_LOG_DIR = 'runs' 