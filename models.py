import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

class Caltech101Classifier(nn.Module):
    """Caltech-101分类器基类"""
    
    def __init__(self, model_name='resnet18', num_classes=101, pretrained=True, freeze_features=False):
        super(Caltech101Classifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_features = freeze_features
        
        # 加载预训练模型
        self.backbone = self._load_backbone()
        
        # 修改分类器层
        self._modify_classifier()
        
        # 冻结特征提取层（如果需要）
        if freeze_features:
            self._freeze_features()
    
    def _load_backbone(self):
        """加载预训练的backbone模型"""
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
        elif self.model_name == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
        elif self.model_name == 'alexnet':
            model = models.alexnet(pretrained=self.pretrained)
        elif self.model_name == 'vgg16':
            model = models.vgg16(pretrained=self.pretrained)
        elif self.model_name == 'densenet121':
            model = models.densenet121(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model
    
    def _modify_classifier(self):
        """修改分类器层以适应Caltech-101的101个类别"""
        if 'resnet' in self.model_name:
            # ResNet系列
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, self.num_classes)
        
        elif self.model_name == 'alexnet':
            # AlexNet
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(num_features, self.num_classes)
        
        elif 'vgg' in self.model_name:
            # VGG系列
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(num_features, self.num_classes)
        
        elif 'densenet' in self.model_name:
            # DenseNet系列
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(num_features, self.num_classes)
    
    def _freeze_features(self):
        """冻结特征提取层的参数"""
        if 'resnet' in self.model_name:
            # 冻结除了最后的全连接层之外的所有层
            for name, param in self.backbone.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
        
        elif self.model_name == 'alexnet':
            # 冻结特征提取部分
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            # 冻结分类器的前几层，只训练最后一层
            for i, param in enumerate(self.backbone.classifier.parameters()):
                if i < 6:  # 前6层冻结
                    param.requires_grad = False
        
        elif 'vgg' in self.model_name:
            # 冻结特征提取部分
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            # 冻结分类器的前几层
            for i, param in enumerate(self.backbone.classifier.parameters()):
                if i < 6:
                    param.requires_grad = False
        
        elif 'densenet' in self.model_name:
            # 冻结特征提取部分
            for name, param in self.backbone.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_trainable_parameters(self):
        """获取可训练的参数"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Model: {self.model_name}")
        print(f"Pretrained: {self.pretrained}")
        print(f"Freeze features: {self.freeze_features}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params:.2%}")

def create_model(model_name='resnet18', num_classes=101, pretrained=True, freeze_features=False):
    """创建模型的工厂函数"""
    model = Caltech101Classifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_features=freeze_features
    )
    return model

def save_model(model, filepath, optimizer=None, epoch=None, best_acc=None, experiment_config=None):
    """保存模型"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model.model_name,
        'num_classes': model.num_classes,
        'pretrained': model.pretrained,
        'freeze_features': model.freeze_features,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if best_acc is not None:
        checkpoint['best_acc'] = best_acc
    
    if experiment_config is not None:
        checkpoint['experiment_config'] = experiment_config
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, device='cpu'):
    """加载模型"""
    checkpoint = torch.load(filepath, map_location=device)
    
    model = Caltech101Classifier(
        model_name=checkpoint['model_name'],
        num_classes=checkpoint['num_classes'],
        pretrained=checkpoint['pretrained'],
        freeze_features=checkpoint['freeze_features']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, checkpoint

def compare_models():
    """比较不同模型的参数量"""
    models_to_compare = ['resnet18', 'resnet50', 'alexnet', 'vgg16', 'densenet121']
    
    print("=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(f"{'Model':<15} {'Total Params':<15} {'Trainable':<15} {'Frozen':<15} {'Ratio':<10}")
    print("-" * 80)
    
    for model_name in models_to_compare:
        try:
            # 预训练 + 微调
            model_finetune = create_model(model_name, pretrained=True, freeze_features=False)
            total_params = sum(p.numel() for p in model_finetune.parameters())
            trainable_params = sum(p.numel() for p in model_finetune.parameters() if p.requires_grad)
            ratio = trainable_params / total_params
            
            print(f"{model_name:<15} {total_params:<15,} {trainable_params:<15,} {total_params-trainable_params:<15,} {ratio:<10.2%}")
            
            # 预训练 + 特征冻结
            model_frozen = create_model(model_name, pretrained=True, freeze_features=True)
            trainable_params_frozen = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
            ratio_frozen = trainable_params_frozen / total_params
            
            print(f"{model_name+' (frozen)':<15} {total_params:<15,} {trainable_params_frozen:<15,} {total_params-trainable_params_frozen:<15,} {ratio_frozen:<10.2%}")
            
        except Exception as e:
            print(f"{model_name:<15} Error: {e}")
    
    print("=" * 80) 