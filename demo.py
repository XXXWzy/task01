#!/usr/bin/env python3
"""
Caltech-101 快速演示脚本
用于快速测试训练功能（使用较少的epochs和较小的batch_size）
"""

import os
import torch
from config import Config
from dataset import get_dataloaders, analyze_dataset
from models import create_model
from trainer import Trainer

def quick_demo():
    """快速演示训练过程"""
    print("Caltech-101 Quick Demo")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs('demo_results', exist_ok=True)
    os.makedirs('demo_models', exist_ok=True)
    os.makedirs('demo_runs', exist_ok=True)
    
    # 加载数据（使用较小的batch_size）
    print("\nLoading data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=Config.DATA_DIR,
        batch_size=16,  # 较小的batch_size
        num_workers=2,  # 较少的workers
        val_split=0.2,
        use_augmentation=True
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 创建模型
    print("\nCreating model...")
    model = create_model(
        model_name='resnet18',
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        freeze_features=False
    )
    
    # 模型信息
    model.print_model_info()
    
    # 快速训练配置
    demo_config = {
        'pretrained': True,
        'freeze_features': False,
        'learning_rate': 0.001,
        'num_epochs': 3,  # 只训练3个epochs
        'lr_scheduler': 'step',
        'step_size': 2,
        'gamma': 0.1
    }
    
    print(f"\nDemo config: {demo_config}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=classes,
        experiment_name='demo_resnet18_quick',
        experiment_config=demo_config,
        device=device
    )
    
    # 训练模型
    print("\nStarting training...")
    best_val_acc = trainer.train()
    
    # 评估模型
    print("\nEvaluating model...")
    test_acc, results = trainer.evaluate_and_save_results()
    
    print(f"\nDemo completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"\nResults saved in demo_results/")
    print(f"Model saved in demo_models/")
    print(f"TensorBoard logs in demo_runs/")
    
    return test_acc

if __name__ == '__main__':
    quick_demo() 