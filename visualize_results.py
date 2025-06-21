#!/usr/bin/env python3
"""
实验结果可视化工具
从TensorBoard日志和结果文件中生成报告图表
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from config import Config

class ExperimentVisualizer:
    """实验结果可视化器"""
    
    def __init__(self, results_dir='results', tensorboard_dir='tensorboard_logs'):
        self.results_dir = results_dir
        self.tensorboard_dir = tensorboard_dir
        self.output_dir = 'visualization_outputs'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置matplotlib中文字体和样式
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
    def extract_tensorboard_data(self, experiment_name):
        """从TensorBoard日志中提取数据"""
        log_dir = os.path.join(self.tensorboard_dir, experiment_name)
        
        if not os.path.exists(log_dir):
            print(f"Warning: TensorBoard log directory not found: {log_dir}")
            return None
            
        try:
            ea = EventAccumulator(log_dir)
            ea.Reload()
            
            # 提取标量数据
            scalar_tags = ea.Tags()['scalars']
            
            data = {}
            for tag in scalar_tags:
                scalar_events = ea.Scalars(tag)
                data[tag] = {
                    'steps': [event.step for event in scalar_events],
                    'values': [event.value for event in scalar_events]
                }
            
            return data
            
        except Exception as e:
            print(f"Error extracting TensorBoard data for {experiment_name}: {e}")
            return None
    
    def plot_training_curves(self, experiment_names=None):
        """绘制训练曲线"""
        if experiment_names is None:
            # 从results目录自动获取实验名称
            result_files = [f for f in os.listdir(self.results_dir) 
                           if f.endswith('_results.json')]
            experiment_names = [f.replace('_results.json', '') for f in result_files]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Progress Comparison', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(experiment_names)))
        
        for idx, exp_name in enumerate(experiment_names):
            color = colors[idx]
            
            # 从JSON结果文件读取数据
            result_file = os.path.join(self.results_dir, f"{exp_name}_results.json")
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                history = data.get('train_history', {})
                
                epochs = list(range(1, len(history.get('train_losses', [])) + 1))
                
                # 训练损失
                if 'train_losses' in history:
                    axes[0, 0].plot(epochs, history['train_losses'], 
                                   color=color, linestyle='-', linewidth=2,
                                   label=f"{exp_name} (Train)", alpha=0.8)
                
                # 验证损失
                if 'val_losses' in history:
                    axes[0, 0].plot(epochs, history['val_losses'], 
                                   color=color, linestyle='--', linewidth=2,
                                   label=f"{exp_name} (Val)", alpha=0.8)
                
                # 训练准确率
                if 'train_accuracies' in history:
                    axes[0, 1].plot(epochs, history['train_accuracies'], 
                                   color=color, linestyle='-', linewidth=2,
                                   label=f"{exp_name} (Train)", alpha=0.8)
                
                # 验证准确率
                if 'val_accuracies' in history:
                    axes[0, 1].plot(epochs, history['val_accuracies'], 
                                   color=color, linestyle='--', linewidth=2,
                                   label=f"{exp_name} (Val)", alpha=0.8)
        
        # 设置子图标题和标签
        axes[0, 0].set_title('Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Accuracy Curves', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 最终性能对比柱状图
        self._plot_final_performance_bars(axes[1, 0], axes[1, 1], experiment_names)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'training_curves_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {output_file}")
        return output_file
    
    def _plot_final_performance_bars(self, ax1, ax2, experiment_names):
        """绘制最终性能对比柱状图"""
        val_accs = []
        test_accs = []
        exp_labels = []
        
        for exp_name in experiment_names:
            result_file = os.path.join(self.results_dir, f"{exp_name}_results.json")
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                val_accs.append(data.get('best_val_acc', 0))
                test_accs.append(data.get('test_acc', 0))
                exp_labels.append(exp_name.replace('_', '\n'))
        
        if val_accs and test_accs:
            x = np.arange(len(exp_labels))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, val_accs, width, label='Validation Acc', 
                           color='skyblue', alpha=0.8)
            bars2 = ax1.bar(x + width/2, test_accs, width, label='Test Acc', 
                           color='lightcoral', alpha=0.8)
            
            ax1.set_title('Final Accuracy Comparison', fontweight='bold')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(exp_labels, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 在柱状图上显示数值
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
            
            for bar in bars2:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 预训练影响分析
        self._plot_pretraining_impact(ax2, experiment_names)
    
    def _plot_pretraining_impact(self, ax, experiment_names):
        """绘制预训练影响分析"""
        pretrained_accs = []
        scratch_accs = []
        
        for exp_name in experiment_names:
            result_file = os.path.join(self.results_dir, f"{exp_name}_results.json")
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                config = data.get('experiment_config', {})
                test_acc = data.get('test_acc', 0)
                
                if config.get('pretrained', False):
                    pretrained_accs.append(test_acc)
                else:
                    scratch_accs.append(test_acc)
        
        if pretrained_accs and scratch_accs:
            categories = ['Pretrained', 'From Scratch']
            means = [np.mean(pretrained_accs), np.mean(scratch_accs)]
            stds = [np.std(pretrained_accs), np.std(scratch_accs)]
            
            bars = ax.bar(categories, means, yerr=stds, capsize=5, 
                         color=['green', 'orange'], alpha=0.7)
            
            ax.set_title('Pretrained vs From Scratch', fontweight='bold')
            ax.set_ylabel('Test Accuracy (%)')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 显示数值
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                ax.text(bar.get_x() + bar.get_width()/2., mean + std + 1,
                       f'{mean:.1f}±{std:.1f}%', ha='center', va='bottom', 
                       fontweight='bold')
            
            # 显示改进幅度
            if len(means) == 2:
                improvement = means[0] - means[1]
                ax.text(0.5, max(means) + max(stds) + 5,
                       f'Improvement: +{improvement:.1f}%', 
                       ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'Insufficient data for\npretraining comparison', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic')
            ax.set_title('Pretrained vs From Scratch', fontweight='bold')
    
    def plot_confusion_matrices(self):
        """绘制混淆矩阵"""
        result_files = [f for f in os.listdir(self.results_dir) 
                       if f.endswith('_results.json')]
        
        for result_file in result_files:
            exp_name = result_file.replace('_results.json', '')
            
            with open(os.path.join(self.results_dir, result_file), 'r') as f:
                data = json.load(f)
            
            # 检查是否有混淆矩阵数据
            if 'confusion_matrix' in data:
                cm = np.array(data['confusion_matrix'])
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
                plt.title(f'Confusion Matrix - {exp_name}', fontweight='bold')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                
                output_file = os.path.join(self.output_dir, f'{exp_name}_confusion_matrix.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Confusion matrix saved: {output_file}")
    
    def generate_performance_summary_plot(self):
        """生成性能总结图"""
        result_files = [f for f in os.listdir(self.results_dir) 
                       if f.endswith('_results.json')]
        
        if not result_files:
            print("No result files found!")
            return
        
        data_rows = []
        
        for result_file in result_files:
            with open(os.path.join(self.results_dir, result_file), 'r') as f:
                data = json.load(f)
            
            exp_name = data['experiment_name']
            config = data.get('experiment_config', {})
            
            # 解析实验信息
            parts = exp_name.split('_')
            model_name = parts[0]
            exp_type = '_'.join(parts[1:])
            
            data_rows.append({
                'Experiment': exp_name,
                'Model': model_name,
                'Type': exp_type,
                'Pretrained': config.get('pretrained', False),
                'Val Acc': data.get('best_val_acc', 0),
                'Test Acc': data.get('test_acc', 0),
                'Epochs': len(data.get('train_history', {}).get('train_losses', [])),
                'LR': config.get('learning_rate', 0)
            })
        
        df = pd.DataFrame(data_rows)
        
        # 创建多子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. 模型对比
        model_performance = df.groupby('Model')['Test Acc'].agg(['mean', 'std']).reset_index()
        axes[0, 0].bar(model_performance['Model'], model_performance['mean'], 
                      yerr=model_performance['std'], capsize=5, alpha=0.7)
        axes[0, 0].set_title('Model Comparison (Test Accuracy)', fontweight='bold')
        axes[0, 0].set_ylabel('Test Accuracy (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. 学习率影响
        lr_performance = df.groupby('LR')['Test Acc'].mean().reset_index()
        axes[0, 1].plot(lr_performance['LR'], lr_performance['Test Acc'], 'o-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Learning Rate Impact', fontweight='bold')
        axes[0, 1].set_xlabel('Learning Rate')
        axes[0, 1].set_ylabel('Test Accuracy (%)')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 训练轮数与性能
        axes[1, 0].scatter(df['Epochs'], df['Test Acc'], c=df['Pretrained'].map({True: 'red', False: 'blue'}), 
                          alpha=0.7, s=100)
        axes[1, 0].set_title('Epochs vs Performance', fontweight='bold')
        axes[1, 0].set_xlabel('Training Epochs')
        axes[1, 0].set_ylabel('Test Accuracy (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加图例
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Pretrained')
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='From Scratch')
        axes[1, 0].legend(handles=[red_patch, blue_patch])
        
        # 4. 验证vs测试准确率
        axes[1, 1].scatter(df['Val Acc'], df['Test Acc'], alpha=0.7, s=100)
        axes[1, 1].plot([df['Val Acc'].min(), df['Val Acc'].max()], 
                       [df['Val Acc'].min(), df['Val Acc'].max()], 'r--', alpha=0.5)
        axes[1, 1].set_title('Validation vs Test Accuracy', fontweight='bold')
        axes[1, 1].set_xlabel('Validation Accuracy (%)')
        axes[1, 1].set_ylabel('Test Accuracy (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'performance_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance analysis saved: {output_file}")
        
        # 保存数据表
        df.to_csv(os.path.join(self.output_dir, 'performance_summary.csv'), index=False)
        print(f"Performance data saved: {os.path.join(self.output_dir, 'performance_summary.csv')}")
        
        return output_file
    
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("Generating comprehensive visualizations...")
        
        outputs = []
        
        # 1. 训练曲线
        print("\n1. Generating training curves...")
        try:
            output = self.plot_training_curves()
            outputs.append(output)
        except Exception as e:
            print(f"Error generating training curves: {e}")
        
        # 2. 性能分析
        print("\n2. Generating performance analysis...")
        try:
            output = self.generate_performance_summary_plot()
            outputs.append(output)
        except Exception as e:
            print(f"Error generating performance analysis: {e}")
        
        # 3. 混淆矩阵
        print("\n3. Generating confusion matrices...")
        try:
            self.plot_confusion_matrices()
        except Exception as e:
            print(f"Error generating confusion matrices: {e}")
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}/")
        return outputs

def main():
    """主函数"""
    print("🎨 Starting experiment visualization...")
    
    visualizer = ExperimentVisualizer()
    outputs = visualizer.generate_all_visualizations()
    
    print(f"\n🎉 Visualization complete!")
    print(f"📁 Output directory: {visualizer.output_dir}/")
    print(f"📊 Generated files:")
    
    for output_file in os.listdir(visualizer.output_dir):
        print(f"   - {output_file}")
    
    print(f"\n📋 Next steps:")
    print(f"1. Review generated charts")
    print(f"2. Include in your report")
    print(f"3. Start TensorBoard for interactive analysis:")
    print(f"   tensorboard --logdir {Config.TENSORBOARD_LOG_DIR}")

if __name__ == '__main__':
    main() 