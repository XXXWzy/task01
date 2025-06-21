import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from config import Config
from models import save_model

class Trainer:
    """训练器类"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, classes, 
                 experiment_name, experiment_config, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.classes = classes
        self.experiment_name = experiment_name
        self.experiment_config = experiment_config
        self.device = device
        
        # 移动模型到设备
        self.model.to(device)
        
        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 设置优化器
        self.optimizer = self._setup_optimizer()
        
        # 设置学习率调度器
        self.scheduler = self._setup_scheduler()
        
        # 设置TensorBoard
        self.writer = SummaryWriter(
            log_dir=os.path.join(Config.TENSORBOARD_LOG_DIR, experiment_name)
        )
        
        # 记录实验配置到TensorBoard
        self._log_experiment_config()
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # 最佳模型
        self.best_val_acc = 0.0
        self.best_model_state = None
        
        # 早停
        self.patience = Config.EARLY_STOPPING_PATIENCE
        self.patience_counter = 0
        
        # 确保输出目录存在
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    def _log_experiment_config(self):
        """记录实验配置到TensorBoard"""
        # 记录超参数
        hparams = {
            'model_name': self.model.model_name,
            'pretrained': self.experiment_config.get('pretrained', False),
            'freeze_features': self.experiment_config.get('freeze_features', False),
            'learning_rate': self.experiment_config.get('learning_rate', 0.001),
            'batch_size': Config.BATCH_SIZE,
            'num_epochs': self.experiment_config.get('num_epochs', 50),
            'lr_scheduler': self.experiment_config.get('lr_scheduler', 'step'),
            'optimizer': 'Adam',
            'weight_decay': 1e-4,
            'loss_function': 'CrossEntropyLoss',
            'train_samples': len(self.train_loader.dataset),
            'val_samples': len(self.val_loader.dataset),
            'test_samples': len(self.test_loader.dataset),
            'num_classes': Config.NUM_CLASSES
        }
        
        # 记录模型参数信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        hparams.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        })
        
        # 将超参数写入TensorBoard
        self.writer.add_hparams(hparams, {})
        
        # 记录文本信息
        config_text = f"""
        ## Experiment Configuration
        
        **Model**: {self.model.model_name}
        **Pretrained**: {self.experiment_config.get('pretrained', False)}
        **Freeze Features**: {self.experiment_config.get('freeze_features', False)}
        **Learning Rate**: {self.experiment_config.get('learning_rate', 0.001)}
        **Batch Size**: {Config.BATCH_SIZE}
        **Epochs**: {self.experiment_config.get('num_epochs', 50)}
        **Optimizer**: Adam
        **Weight Decay**: 1e-4
        **LR Scheduler**: {self.experiment_config.get('lr_scheduler', 'step')}
        
        **Dataset Split**:
        - Training: {len(self.train_loader.dataset)} samples
        - Validation: {len(self.val_loader.dataset)} samples  
        - Testing: {len(self.test_loader.dataset)} samples
        
        **Model Parameters**:
        - Total: {total_params:,}
        - Trainable: {trainable_params:,}
        - Frozen: {total_params - trainable_params:,}
        """
        
        self.writer.add_text("Experiment_Config", config_text, 0)

    def _setup_optimizer(self):
        """设置优化器"""
        # 获取可训练参数
        trainable_params = self.model.get_trainable_parameters()
        
        # 如果是微调，可以为不同层设置不同的学习率
        if self.experiment_config.get('freeze_features', False):
            # 特征冻结：只训练分类器
            optimizer = optim.Adam(
                trainable_params,
                lr=self.experiment_config['learning_rate'],
                weight_decay=1e-4
            )
        else:
            # 微调：为backbone和分类器设置不同的学习率
            if self.experiment_config.get('pretrained', False):
                # 预训练模型微调
                backbone_params = []
                classifier_params = []
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        if 'fc' in name or 'classifier' in name:
                            classifier_params.append(param)
                        else:
                            backbone_params.append(param)
                
                optimizer = optim.Adam([
                    {'params': backbone_params, 'lr': self.experiment_config['learning_rate'] * 0.1},
                    {'params': classifier_params, 'lr': self.experiment_config['learning_rate']}
                ], weight_decay=1e-4)
            else:
                # 从头训练
                optimizer = optim.Adam(
                    trainable_params,
                    lr=self.experiment_config['learning_rate'],
                    weight_decay=1e-4
                )
        
        return optimizer
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        if self.experiment_config['lr_scheduler'] == 'step':
            scheduler = StepLR(
                self.optimizer, 
                step_size=self.experiment_config['step_size'], 
                gamma=self.experiment_config['gamma']
            )
        elif self.experiment_config['lr_scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.experiment_config['num_epochs']
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        # 计算精确度、召回率、F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='macro', zero_division=0
        )
        
        return epoch_loss, epoch_acc, precision, recall, f1
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        # 计算精确度、召回率、F1分数（验证集上的mAP替代指标）
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='macro', zero_division=0
        )
        
        return epoch_loss, epoch_acc, precision, recall, f1
    
    def test(self):
        """测试模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        test_loss = running_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc, all_predictions, all_targets
    
    def train(self):
        """完整训练过程"""
        print(f"\nStarting training for experiment: {self.experiment_name}")
        print("=" * 50)
        
        # 打印模型信息
        self.model.print_model_info()
        print("=" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.experiment_config['num_epochs']):
            print(f'\nEpoch {epoch+1}/{self.experiment_config["num_epochs"]}')
            print('-' * 30)
            
            # 训练
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.train_epoch()
            
            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate_epoch()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.experiment_config['learning_rate']
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)
            
            # TensorBoard记录 - 详细记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Precision/Train', train_precision, epoch)
            self.writer.add_scalar('Precision/Validation', val_precision, epoch)
            self.writer.add_scalar('Recall/Train', train_recall, epoch)
            self.writer.add_scalar('Recall/Validation', val_recall, epoch)
            self.writer.add_scalar('F1/Train', train_f1, epoch)
            self.writer.add_scalar('F1/Validation', val_f1, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # 记录损失比较
            self.writer.add_scalars('Loss_Comparison', {
                'Train': train_loss,
                'Validation': val_loss
            }, epoch)
            
            # 记录准确率比较
            self.writer.add_scalars('Accuracy_Comparison', {
                'Train': train_acc,
                'Validation': val_acc
            }, epoch)
            
            # 记录验证集详细指标（类似mAP的综合指标）
            mean_average_precision = (val_precision + val_recall + val_f1) / 3
            self.writer.add_scalar('Validation/Mean_Average_Metrics', mean_average_precision, epoch)
            
            # 打印结果
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                
                # 保存最佳模型
                best_model_path = os.path.join(
                    Config.MODEL_DIR, 
                    f'{self.experiment_name}_best.pth'
                )
                save_model(
                    self.model, 
                    best_model_path, 
                    self.optimizer, 
                    epoch, 
                    self.best_val_acc,
                    self.experiment_config
                )
                print(f'New best model saved! Val Acc: {val_acc:.2f}%')
            else:
                self.patience_counter += 1
            
            # 早停检查
            if self.patience_counter >= self.patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # 训练结束
        training_time = time.time() - start_time
        print(f'\nTraining completed in {training_time//60:.0f}m {training_time%60:.0f}s')
        print(f'Best validation accuracy: {self.best_val_acc:.2f}%')
        
        # 恢复最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # 记录最终结果到TensorBoard
        self.writer.add_scalar('Final/Best_Validation_Accuracy', self.best_val_acc, 0)
        self.writer.add_scalar('Final/Training_Time_Minutes', training_time/60, 0)
        
        # 关闭TensorBoard writer
        self.writer.close()
        
        return self.best_val_acc
    
    def evaluate_and_save_results(self):
        """评估模型并保存结果"""
        print("\nEvaluating on test set...")
        test_loss, test_acc, predictions, targets = self.test()
        
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_acc:.2f}%')
        
        # 生成分类报告
        class_names = [cls.replace('_', ' ') for cls in self.classes]
        report = classification_report(
            targets, predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # 保存结果
        results = {
            'experiment_name': self.experiment_name,
            'experiment_config': self.experiment_config,
            'best_val_acc': self.best_val_acc,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'train_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies
            },
            'classification_report': report
        }
        
        # 保存结果到文件
        import json
        results_path = os.path.join(Config.OUTPUT_DIR, f'{self.experiment_name}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 生成混淆矩阵
        self._plot_confusion_matrix(targets, predictions, class_names)
        
        # 生成训练曲线
        self._plot_training_curves()
        
        return test_acc, results
    
    def _plot_confusion_matrix(self, targets, predictions, class_names):
        """绘制混淆矩阵"""
        # 计算混淆矩阵
        cm = confusion_matrix(targets, predictions)
        
        # 只显示前20个类别的混淆矩阵（避免过于拥挤）
        if len(class_names) > 20:
            top_classes = np.argsort(np.diag(cm))[-20:]  # 选择准确率最高的20个类别
            cm_subset = cm[np.ix_(top_classes, top_classes)]
            class_names_subset = [class_names[i] for i in top_classes]
        else:
            cm_subset = cm
            class_names_subset = class_names
        
        # 绘制混淆矩阵
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names_subset,
                   yticklabels=class_names_subset)
        plt.title(f'Confusion Matrix - {self.experiment_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
        cm_path = os.path.join(Config.OUTPUT_DIR, f'{self.experiment_name}_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Confusion matrix saved to {cm_path}')
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        curves_path = os.path.join(Config.OUTPUT_DIR, f'{self.experiment_name}_training_curves.png')
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Training curves saved to {curves_path}')

class EarlyStopping:
    """早停类"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience 