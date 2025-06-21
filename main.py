#!/usr/bin/env python3
"""
Caltech-101 图像分类主训练脚本
使用预训练的CNN网络进行微调
"""

import os
import sys
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import json

from config import Config
from dataset import get_dataloaders, analyze_dataset
from models import create_model, compare_models
from trainer import Trainer

def set_seed(seed=42):
    """设置随机种子以确保实验可重复"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    
    return device

def run_single_experiment(model_name, experiment_name, experiment_config, 
                         train_loader, val_loader, test_loader, classes, device):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Config: {experiment_config}")
    print(f"{'='*80}")
    
    # 创建模型
    model = create_model(
        model_name=model_name,
        num_classes=Config.NUM_CLASSES,
        pretrained=experiment_config['pretrained'],
        freeze_features=experiment_config['freeze_features']
    )
    
    # 创建完整的实验名称
    full_experiment_name = f"{model_name}_{experiment_name}"
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=classes,
        experiment_name=full_experiment_name,
        experiment_config=experiment_config,
        device=device
    )
    
    # 训练模型
    best_val_acc = trainer.train()
    
    # 评估并保存结果
    test_acc, results = trainer.evaluate_and_save_results()
    
    print(f"\nExperiment {full_experiment_name} completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    
    return test_acc, results

def run_all_experiments(models_to_test=['resnet18'], experiments_to_run=None):
    """运行所有实验"""
    print("Caltech-101 Classification Experiments")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 设置设备
    device = setup_device()
    
    # 分析数据集
    print("\nAnalyzing dataset...")
    class_stats = analyze_dataset(Config.DATA_DIR)
    
    # 加载数据
    print("\nLoading data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        val_split=Config.VAL_SPLIT,
        use_augmentation=Config.USE_AUGMENTATION
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 比较模型
    print("\nComparing models...")
    compare_models()
    
    # 如果没有指定实验，运行所有实验
    if experiments_to_run is None:
        experiments_to_run = list(Config.EXPERIMENTS.keys())
    
    # 存储所有结果
    all_results = {}
    
    # 为每个模型运行每个实验
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"{'='*60}")
        
        model_results = {}
        
        for experiment_name in experiments_to_run:
            if experiment_name not in Config.EXPERIMENTS:
                print(f"Warning: Unknown experiment {experiment_name}")
                continue
            
            experiment_config = Config.EXPERIMENTS[experiment_name]
            
            try:
                test_acc, results = run_single_experiment(
                    model_name=model_name,
                    experiment_name=experiment_name,
                    experiment_config=experiment_config,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    classes=classes,
                    device=device
                )
                
                model_results[experiment_name] = {
                    'test_acc': test_acc,
                    'results': results
                }
                
            except Exception as e:
                print(f"Error in experiment {model_name}_{experiment_name}: {e}")
                continue
        
        all_results[model_name] = model_results
    
    # 生成综合报告
    generate_comparison_report(all_results)
    
    return all_results

def generate_comparison_report(all_results):
    """生成实验对比报告"""
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON REPORT")
    print("="*80)
    
    # 创建结果表格
    results_table = []
    
    for model_name, model_results in all_results.items():
        for experiment_name, experiment_data in model_results.items():
            results_table.append({
                'Model': model_name,
                'Experiment': experiment_name,
                'Test Accuracy (%)': experiment_data['test_acc'],
                'Pretrained': experiment_data['results']['experiment_config']['pretrained'],
                'Freeze Features': experiment_data['results']['experiment_config']['freeze_features'],
                'Learning Rate': experiment_data['results']['experiment_config']['learning_rate'],
                'Epochs': len(experiment_data['results']['train_history']['train_losses'])
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(results_table)
    
    # 保存结果表格
    results_path = os.path.join(Config.OUTPUT_DIR, 'comparison_results.csv')
    df.to_csv(results_path, index=False)
    print(f"\nResults table saved to: {results_path}")
    
    # 打印结果表格
    print("\nResults Summary:")
    print("-" * 80)
    print(df.to_string(index=False))
    
    # 生成可视化图表
    generate_comparison_plots(df, all_results)
    
    # 分析预训练的影响
    analyze_pretraining_impact(df)

def generate_comparison_plots(df, all_results):
    """生成对比图表"""
    
    # 1. 测试准确率对比图
    plt.figure(figsize=(12, 8))
    
    # 按模型和实验分组
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        plt.bar([f"{model}\n{exp}" for exp in model_data['Experiment']], 
               model_data['Test Accuracy (%)'], 
               alpha=0.7, 
               label=model)
    
    plt.title('Test Accuracy Comparison Across Models and Experiments')
    plt.xlabel('Model + Experiment')
    plt.ylabel('Test Accuracy (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    accuracy_plot_path = os.path.join(Config.OUTPUT_DIR, 'accuracy_comparison.png')
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy comparison plot saved to: {accuracy_plot_path}")
    
    # 2. 预训练 vs 从头训练对比
    plt.figure(figsize=(10, 6))
    
    pretrained_results = df[df['Pretrained'] == True]['Test Accuracy (%)']
    scratch_results = df[df['Pretrained'] == False]['Test Accuracy (%)']
    
    data_to_plot = []
    labels = []
    
    if len(pretrained_results) > 0:
        data_to_plot.append(pretrained_results)
        labels.append('Pre-trained')
    
    if len(scratch_results) > 0:
        data_to_plot.append(scratch_results)
        labels.append('From Scratch')
    
    if data_to_plot:
        plt.boxplot(data_to_plot, labels=labels)
        plt.title('Pre-trained vs From Scratch Performance')
        plt.ylabel('Test Accuracy (%)')
        plt.grid(True, alpha=0.3)
        
        pretraining_plot_path = os.path.join(Config.OUTPUT_DIR, 'pretraining_comparison.png')
        plt.savefig(pretraining_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Pre-training comparison plot saved to: {pretraining_plot_path}")
    
    # 3. 训练曲线对比
    plt.figure(figsize=(15, 10))
    
    subplot_idx = 1
    for model_name, model_results in all_results.items():
        for experiment_name, experiment_data in model_results.items():
            plt.subplot(2, 2, subplot_idx)
            
            train_history = experiment_data['results']['train_history']
            epochs = range(1, len(train_history['train_losses']) + 1)
            
            plt.plot(epochs, train_history['train_accuracies'], 'b-', 
                    label='Training', alpha=0.7)
            plt.plot(epochs, train_history['val_accuracies'], 'r-', 
                    label='Validation', alpha=0.7)
            
            plt.title(f'{model_name} - {experiment_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            subplot_idx += 1
            if subplot_idx > 4:
                break
        if subplot_idx > 4:
            break
    
    plt.tight_layout()
    curves_comparison_path = os.path.join(Config.OUTPUT_DIR, 'training_curves_comparison.png')
    plt.savefig(curves_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves comparison saved to: {curves_comparison_path}")

def analyze_pretraining_impact(df):
    """分析预训练的影响"""
    print("\n" + "="*60)
    print("PRETRAINING IMPACT ANALYSIS")
    print("="*60)
    
    pretrained_results = df[df['Pretrained'] == True]
    scratch_results = df[df['Pretrained'] == False]
    
    if len(pretrained_results) > 0 and len(scratch_results) > 0:
        pretrained_mean = pretrained_results['Test Accuracy (%)'].mean()
        scratch_mean = scratch_results['Test Accuracy (%)'].mean()
        improvement = pretrained_mean - scratch_mean
        
        print(f"Average accuracy with pre-training: {pretrained_mean:.2f}%")
        print(f"Average accuracy from scratch: {scratch_mean:.2f}%")
        print(f"Improvement from pre-training: {improvement:.2f} percentage points")
        print(f"Relative improvement: {improvement/scratch_mean*100:.1f}%")
        
        # 最佳结果对比
        best_pretrained = pretrained_results['Test Accuracy (%)'].max()
        best_scratch = scratch_results['Test Accuracy (%)'].max()
        
        print(f"\nBest accuracy with pre-training: {best_pretrained:.2f}%")
        print(f"Best accuracy from scratch: {best_scratch:.2f}%")
        print(f"Best improvement: {best_pretrained - best_scratch:.2f} percentage points")
    
    else:
        print("Not enough data for pre-training comparison")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Caltech-101 Classification Experiments')
    parser.add_argument('--models', nargs='+', default=['resnet18'], 
                       choices=['resnet18', 'resnet50', 'alexnet', 'vgg16', 'densenet121'],
                       help='Models to test')
    parser.add_argument('--experiments', nargs='+', 
                       choices=list(Config.EXPERIMENTS.keys()),
                       help='Experiments to run (default: all)')
    parser.add_argument('--data-dir', default=Config.DATA_DIR,
                       help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=Config.NUM_WORKERS,
                       help='Number of data loading workers')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze dataset without training')
    
    args = parser.parse_args()
    
    # 更新配置
    Config.DATA_DIR = args.data_dir
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_WORKERS = args.num_workers
    
    # 创建输出目录
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.TENSORBOARD_LOG_DIR, exist_ok=True)
    
    if args.analyze_only:
        # 只分析数据集
        analyze_dataset(Config.DATA_DIR)
        return
    
    # 运行实验
    results = run_all_experiments(
        models_to_test=args.models,
        experiments_to_run=args.experiments
    )
    
    print(f"\nAll experiments completed!")
    print(f"Results saved in: {Config.OUTPUT_DIR}")
    print(f"Models saved in: {Config.MODEL_DIR}")
    print(f"TensorBoard logs in: {Config.TENSORBOARD_LOG_DIR}")
    print(f"\nTo view TensorBoard logs, run:")
    print(f"tensorboard --logdir {Config.TENSORBOARD_LOG_DIR}")

if __name__ == '__main__':
    main() 