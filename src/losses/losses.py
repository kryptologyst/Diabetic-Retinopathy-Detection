"""Loss functions for diabetic retinopathy detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss = -alpha * (1-p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (if None, uses class frequency)
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLossAlpha(nn.Module):
    """Focal Loss with learnable alpha parameters."""
    
    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """Initialize Focal Loss with learnable alpha.
        
        Args:
            num_classes: Number of classes
            gamma: Focusing parameter
            reduction: Reduction method
            ignore_index: Index to ignore
        """
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # Learnable alpha parameters
        self.alpha = nn.Parameter(torch.ones(num_classes))
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss with learnable alpha.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        
        # Get alpha for each target
        alpha_t = self.alpha[targets]
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss."""
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """Initialize Label Smoothing Cross Entropy.
        
        Args:
            smoothing: Label smoothing factor
            reduction: Reduction method
            ignore_index: Index to ignore
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing cross entropy.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Label smoothing cross entropy loss
        """
        log_preds = F.log_softmax(inputs, dim=1)
        
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            targets = targets * mask.long()
        
        # Create smoothed targets
        num_classes = inputs.size(1)
        smooth_targets = torch.zeros_like(log_preds)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        loss = -smooth_targets * log_preds
        loss = loss.sum(dim=1)
        
        if self.ignore_index >= 0:
            loss = loss * mask.float()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = 'mean'
    ):
        """Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Dice loss
        """
        # Convert to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot
        targets = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
        # Compute Dice coefficient for each class
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """Combined loss function for diabetic retinopathy detection."""
    
    def __init__(
        self,
        num_classes: int = 2,
        focal_alpha: Optional[float] = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        class_weights: Optional[List[float]] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize combined loss.
        
        Args:
            num_classes: Number of classes
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            label_smoothing: Label smoothing factor
            class_weights: Class weights for cross entropy
            loss_weights: Weights for different loss components
        """
        super().__init__()
        
        if loss_weights is None:
            loss_weights = {'focal': 1.0, 'ce': 0.0, 'smoothing': 0.0}
        
        self.loss_weights = loss_weights
        
        # Initialize loss functions
        if loss_weights['focal'] > 0:
            if focal_alpha is not None:
                self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            else:
                self.focal_loss = FocalLossAlpha(num_classes=num_classes, gamma=focal_gamma)
        
        if loss_weights['ce'] > 0:
            if class_weights is not None:
                class_weights = torch.FloatTensor(class_weights)
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        if loss_weights['smoothing'] > 0:
            self.smoothing_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Combined loss
        """
        total_loss = 0.0
        
        if self.loss_weights['focal'] > 0:
            focal_loss = self.focal_loss(inputs, targets)
            total_loss += self.loss_weights['focal'] * focal_loss
        
        if self.loss_weights['ce'] > 0:
            ce_loss = self.ce_loss(inputs, targets)
            total_loss += self.loss_weights['ce'] * ce_loss
        
        if self.loss_weights['smoothing'] > 0:
            smoothing_loss = self.smoothing_loss(inputs, targets)
            total_loss += self.loss_weights['smoothing'] * smoothing_loss
        
        return total_loss


def create_loss_function(config: dict) -> nn.Module:
    """Create loss function from configuration.
    
    Args:
        config: Loss configuration
        
    Returns:
        Loss function
    """
    loss_name = config['loss']['name']
    
    if loss_name == 'cross_entropy':
        class_weights = config['loss'].get('class_weights', None)
        if class_weights:
            class_weights = torch.FloatTensor(class_weights)
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_name == 'focal':
        return FocalLoss(
            alpha=config['loss'].get('focal_alpha', None),
            gamma=config['loss'].get('focal_gamma', 2.0)
        )
    
    elif loss_name == 'focal_alpha':
        return FocalLossAlpha(
            num_classes=config['data']['num_classes'],
            gamma=config['loss'].get('focal_gamma', 2.0)
        )
    
    elif loss_name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(
            smoothing=config['loss'].get('label_smoothing', 0.1)
        )
    
    elif loss_name == 'combined':
        return CombinedLoss(
            num_classes=config['data']['num_classes'],
            focal_alpha=config['loss'].get('focal_alpha', None),
            focal_gamma=config['loss'].get('focal_gamma', 2.0),
            label_smoothing=config['loss'].get('label_smoothing', 0.1),
            class_weights=config['loss'].get('class_weights', None),
            loss_weights=config['loss'].get('loss_weights', None)
        )
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
