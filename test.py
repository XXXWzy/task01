#!/usr/bin/env python3
"""
Caltech-101 测试脚本
用于测试已训练的模型
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders, get_transforms
from models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def test_model(model_path, data_dir=None, batch_size=32, device='cpu'):
    """测试单个模型"""
    print(f"Loading model from: {model_path}")
    
    # 加载模型
    model, checkpoint = load_model(model_path, device)
    model.eval()
    
    print(f"Model: {checkpoint['model_name']}")
    print(f"Pretrained: {checkpoint['pretrained']}")
    print(f"Freeze features: {checkpoint['freeze_features']}")
    if 'best_acc' in checkpoint:
        print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")
    
    # 加载数据
    if data_dir is None:
        data_dir = Config.DATA_DIR
    
    _, _, test_loader, classes = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        val_split=Config.VAL_SPLIT,
        use_augmentation=False
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 测试模型
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 获取预测结果
            probabilities = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy, all_predictions, all_targets, all_probabilities, classes

def analyze_predictions(predictions, targets, probabilities, classes, save_dir=None):
    """分析预测结果"""
    
    # 生成分类报告
    class_names = [cls.replace('_', ' ') for cls in classes]
    report = classification_report(
        targets, predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    print("\nClassification Report:")
    print(classification_report(targets, predictions, target_names=class_names))
    
    # 计算混淆矩阵
    cm = confusion_matrix(targets, predictions)
    
    # 找出表现最好和最差的类别
    class_accuracies = []
    for i in range(len(classes)):
        if np.sum(cm[i, :]) > 0:  # 避免除零
            acc = cm[i, i] / np.sum(cm[i, :])
            class_accuracies.append((i, classes[i], acc))
    
    class_accuracies.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 10 Best Performing Classes:")
    for i, (idx, class_name, acc) in enumerate(class_accuracies[:10]):
        print(f"{i+1:2d}. {class_name:<20} {acc*100:.2f}%")
    
    print("\nTop 10 Worst Performing Classes:")
    for i, (idx, class_name, acc) in enumerate(class_accuracies[-10:]):
        print(f"{i+1:2d}. {class_name:<20} {acc*100:.2f}%")
    
    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存分类报告
        with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # 绘制混淆矩阵（只显示前20个类别）
        if len(classes) > 20:
            top_classes = [x[0] for x in class_accuracies[:20]]
            cm_subset = cm[np.ix_(top_classes, top_classes)]
            class_names_subset = [classes[i] for i in top_classes]
        else:
            cm_subset = cm
            class_names_subset = class_names
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names_subset,
                   yticklabels=class_names_subset)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制类别准确率分布
        accuracies = [x[2] for x in class_accuracies]
        plt.figure(figsize=(10, 6))
        plt.hist(accuracies, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Accuracy')
        plt.ylabel('Number of Classes')
        plt.title('Distribution of Class Accuracies')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'accuracy_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis results saved to: {save_dir}")
    
    return report, class_accuracies

def predict_single_image(model_path, image_path, device='cpu'):
    """预测单张图像"""
    
    # 加载模型
    model, checkpoint = load_model(model_path, device)
    model.eval()
    
    # 加载类别名称
    _, _, _, classes = get_dataloaders(
        data_dir=Config.DATA_DIR,
        batch_size=1,
        num_workers=1,
        val_split=Config.VAL_SPLIT,
        use_augmentation=False
    )
    
    # 加载和预处理图像
    transform = get_transforms('test', use_augmentation=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = output.max(1)
    
    # 获取top-5预测
    top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
    
    print(f"Image: {image_path}")
    print("Top-5 Predictions:")
    for i in range(5):
        class_idx = top5_indices[0][i].item()
        prob = top5_probs[0][i].item()
        class_name = classes[class_idx]
        print(f"{i+1}. {class_name:<20} {prob*100:.2f}%")
    
    return predicted.item(), probabilities[0].cpu().numpy()

def compare_models(model_paths, data_dir=None, device='cpu'):
    """比较多个模型的性能"""
    results = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.pth', '')
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"{'='*60}")
        
        try:
            accuracy, predictions, targets, probabilities, classes = test_model(
                model_path, data_dir, device=device
            )
            results[model_name] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'targets': targets,
                'probabilities': probabilities
            }
        except Exception as e:
            print(f"Error testing model {model_name}: {e}")
            continue
    
    # 比较结果
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Test Accuracy':<15}")
    print("-" * 45)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for model_name, result in sorted_results:
        print(f"{model_name:<30} {result['accuracy']:.2f}%")
    
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Test Caltech-101 Classification Models')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--models', nargs='+', help='Paths to multiple model files')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--data-dir', default=Config.DATA_DIR, help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for results')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], 
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.image:
        # 预测单张图像
        if not args.model:
            print("Error: --model is required for single image prediction")
            return
        
        predict_single_image(args.model, args.image, device)
    
    elif args.models:
        # 比较多个模型
        results = compare_models(args.models, args.data_dir, device)
        
        # 保存比较结果
        comparison_file = os.path.join(args.output_dir, 'model_comparison.json')
        with open(comparison_file, 'w') as f:
            # 只保存准确率，因为其他数据太大
            comparison_data = {name: {'accuracy': result['accuracy']} 
                             for name, result in results.items()}
            json.dump(comparison_data, f, indent=2)
        print(f"Comparison results saved to: {comparison_file}")
    
    elif args.model:
        # 测试单个模型
        accuracy, predictions, targets, probabilities, classes = test_model(
            args.model, args.data_dir, args.batch_size, device
        )
        
        # 分析预测结果
        model_name = os.path.basename(args.model).replace('.pth', '')
        save_dir = os.path.join(args.output_dir, model_name)
        report, class_accuracies = analyze_predictions(
            predictions, targets, probabilities, classes, save_dir
        )
    
    else:
        print("Error: Please provide --model, --models, or --image argument")

if __name__ == '__main__':
    main() 