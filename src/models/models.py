"""Advanced models for diabetic retinopathy detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, Any
import timm
import logging


class DRClassifier(nn.Module):
    """Base diabetic retinopathy classifier with multiple architecture support."""
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False
    ):
        """Initialize DR classifier.
        
        Args:
            model_name: Name of the backbone model
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Create backbone
        self.backbone = self._create_backbone(model_name, pretrained)
        
        # Get feature dimension
        if hasattr(self.backbone, 'classifier'):
            # EfficientNet, DenseNet
            feature_dim = self.backbone.classifier.in_features
        elif hasattr(self.backbone, 'fc'):
            # ResNet, VGG
            feature_dim = self.backbone.fc.in_features
        elif hasattr(self.backbone, 'head'):
            # Vision Transformer
            feature_dim = self.backbone.head.in_features
        else:
            raise ValueError(f"Unknown backbone architecture: {model_name}")
        
        # Replace classifier
        self._replace_classifier(feature_dim, num_classes, dropout)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _create_backbone(self, model_name: str, pretrained: bool) -> nn.Module:
        """Create backbone model.
        
        Args:
            model_name: Name of the model
            pretrained: Whether to use pretrained weights
            
        Returns:
            Backbone model
        """
        if model_name.startswith('resnet'):
            return models.__dict__[model_name](pretrained=pretrained)
        elif model_name.startswith('efficientnet'):
            return timm.create_model(model_name, pretrained=pretrained)
        elif model_name.startswith('vit_'):
            return timm.create_model(model_name, pretrained=pretrained)
        elif model_name.startswith('convnext'):
            return timm.create_model(model_name, pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _replace_classifier(self, feature_dim: int, num_classes: int, dropout: float) -> None:
        """Replace the classifier head.
        
        Args:
            feature_dim: Input feature dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        if hasattr(self.backbone, 'classifier'):
            # EfficientNet, DenseNet
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes)
            )
        elif hasattr(self.backbone, 'fc'):
            # ResNet, VGG
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes)
            )
        elif hasattr(self.backbone, 'head'):
            # Vision Transformer
            self.backbone.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes)
            )
    
    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        return self.backbone(x)


class AttentionDRClassifier(nn.Module):
    """DR classifier with attention mechanism for interpretability."""
    
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        attention_dim: int = 256
    ):
        """Initialize attention-based DR classifier.
        
        Args:
            backbone_name: Name of the backbone model
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
            attention_dim: Attention dimension
        """
        super().__init__()
        
        # Create backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        
        # Get feature dimensions
        feature_info = self.backbone.feature_info
        self.feature_dim = feature_info[-1]['num_chs']  # Last feature map channels
        self.spatial_dim = feature_info[-1]['reduction']  # Spatial reduction factor
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(self.feature_dim, attention_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass with attention.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (logits, attention_map)
        """
        # Extract features
        features = self.backbone(x)[-1]  # Last feature map
        
        # Compute attention
        attention_map = self.attention(features)
        
        # Apply attention
        attended_features = features * attention_map
        
        # Global pooling
        pooled_features = self.global_pool(attended_features).flatten(1)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits, attention_map


class MultiScaleDRClassifier(nn.Module):
    """DR classifier with multi-scale feature fusion."""
    
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        """Initialize multi-scale DR classifier.
        
        Args:
            backbone_name: Name of the backbone model
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()
        
        # Create backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        
        # Get feature dimensions
        feature_info = self.backbone.feature_info
        self.feature_dims = [f['num_chs'] for f in feature_info]
        
        # Multi-scale fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(self.feature_dims), 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale fusion.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Resize all features to the same spatial size
        target_size = features[-1].shape[-2:]
        resized_features = []
        
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(feat)
        
        # Concatenate features
        fused_features = torch.cat(resized_features, dim=1)
        
        # Fusion convolution
        fused_features = self.fusion_conv(fused_features)
        
        # Global pooling
        pooled_features = self.global_pool(fused_features).flatten(1)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits


class EnsembleDRClassifier(nn.Module):
    """Ensemble of multiple DR classifiers."""
    
    def __init__(
        self,
        model_configs: list,
        num_classes: int = 2,
        ensemble_method: str = "average"
    ):
        """Initialize ensemble classifier.
        
        Args:
            model_configs: List of model configurations
            num_classes: Number of output classes
            ensemble_method: Ensemble method ('average', 'weighted', 'voting')
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.ensemble_method = ensemble_method
        
        # Create individual models
        self.models = nn.ModuleList()
        for config in model_configs:
            model = DRClassifier(**config)
            self.models.append(model)
        
        # Learnable weights for weighted ensemble
        if ensemble_method == "weighted":
            self.weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble output logits
        """
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        outputs = torch.stack(outputs, dim=0)  # [num_models, batch_size, num_classes]
        
        if self.ensemble_method == "average":
            return outputs.mean(dim=0)
        elif self.ensemble_method == "weighted":
            weights = F.softmax(self.weights, dim=0)
            weights = weights.view(-1, 1, 1)  # [num_models, 1, 1]
            return (outputs * weights).sum(dim=0)
        elif self.ensemble_method == "voting":
            # Hard voting
            predictions = outputs.argmax(dim=-1)  # [num_models, batch_size]
            return F.one_hot(predictions.mode(dim=0)[0], num_classes=self.num_classes).float()
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")


def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        PyTorch model
    """
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    pretrained = config['model']['pretrained']
    dropout = config['model']['dropout']
    
    if model_name == "attention_efficientnet":
        return AttentionDRClassifier(
            backbone_name="efficientnet_b0",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_name == "multiscale_efficientnet":
        return MultiScaleDRClassifier(
            backbone_name="efficientnet_b0",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_name == "ensemble":
        model_configs = [
            {"model_name": "efficientnet_b0", "num_classes": num_classes, "pretrained": pretrained, "dropout": dropout},
            {"model_name": "resnet18", "num_classes": num_classes, "pretrained": pretrained, "dropout": dropout},
            {"model_name": "vit_base_patch16_224", "num_classes": num_classes, "pretrained": pretrained, "dropout": dropout}
        ]
        return EnsembleDRClassifier(
            model_configs=model_configs,
            num_classes=num_classes,
            ensemble_method="weighted"
        )
    else:
        return DRClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
