"""Training module for diabetic retinopathy detection."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import time
from tqdm import tqdm

from ..utils.utils import EarlyStopping, save_checkpoint, load_checkpoint
from ..metrics.metrics import MetricsCalculator, CalibrationCalculator, MetricsVisualizer


class Trainer:
    """Trainer class for diabetic retinopathy detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
            device: Device to train on
            logger: Logger instance
        """
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_criterion()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            num_classes=config['data']['num_classes'],
            class_names=config['data']['class_names']
        )
        
        # Initialize calibration calculator
        self.calibration_calculator = CalibrationCalculator()
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['patience'],
            restore_best_weights=True
        )
        
        # Mixed precision training
        self.use_amp = config['device']['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_config = self.config['training']
        
        if optimizer_config.get('optimizer', 'adam').lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_config.get('optimizer', 'adam').lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config.get('optimizer', 'adam')}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_config = self.config['training'].get('scheduler', 'cosine')
        
        if scheduler_config == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        elif scheduler_config == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['num_epochs'] // 3,
                gamma=0.1
            )
        elif scheduler_config == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function."""
        from ..losses.losses import create_loss_function
        return create_loss_function(self.config)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Update loss
            total_loss += loss.item()
            
            # Store predictions and targets
            predictions = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average loss and metrics
        avg_loss = total_loss / len(train_loader)
        
        # Convert to tensors for metrics calculation
        all_predictions = torch.tensor(all_predictions)
        all_targets = torch.tensor(all_targets)
        all_probabilities = torch.tensor(all_probabilities)
        
        metrics = self.metrics_calculator.calculate_metrics(
            all_predictions, all_targets, all_probabilities
        )
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Store predictions and targets
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate average loss and metrics
        avg_loss = total_loss / len(val_loader)
        
        # Convert to tensors for metrics calculation
        all_predictions = torch.tensor(all_predictions)
        all_targets = torch.tensor(all_targets)
        all_probabilities = torch.tensor(all_probabilities)
        
        metrics = self.metrics_calculator.calculate_metrics(
            all_predictions, all_targets, all_probabilities
        )
        
        # Calculate calibration metrics
        calibration_metrics = self.calibration_calculator.calculate_calibration_metrics(
            all_probabilities, all_targets
        )
        metrics.update(calibration_metrics)
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        num_epochs = self.config['training']['num_epochs']
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)
            
            # Log metrics
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            self.logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}, "
                           f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            if 'auroc' in val_metrics:
                self.logger.info(f"Val AUROC: {val_metrics['auroc']:.4f}")
            
            # Save checkpoint
            if save_dir:
                is_best = self._is_best_metrics(val_metrics)
                
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, val_metrics,
                    checkpoint_path, is_best
                )
                
                if is_best:
                    best_path = save_dir / "best_model.pth"
                    save_checkpoint(
                        self.model, self.optimizer, epoch, val_loss, val_metrics,
                        best_path, is_best
                    )
                    self.best_metrics = val_metrics.copy()
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return self.training_history
    
    def _is_best_metrics(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics are the best so far.
        
        Args:
            metrics: Current validation metrics
            
        Returns:
            True if current metrics are the best
        """
        if not self.best_metrics:
            return True
        
        # Use AUROC as primary metric, fallback to accuracy
        if 'auroc' in metrics and 'auroc' in self.best_metrics:
            return metrics['auroc'] > self.best_metrics['auroc']
        else:
            return metrics['accuracy'] > self.best_metrics['accuracy']
    
    def evaluate(
        self,
        test_loader: DataLoader,
        save_dir: Optional[Path] = None
    ) -> Dict[str, float]:
        """Evaluate the model on test set.
        
        Args:
            test_loader: Test data loader
            save_dir: Directory to save evaluation results
            
        Returns:
            Test metrics dictionary
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Testing"):
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                # Store predictions and targets
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to tensors for metrics calculation
        all_predictions = torch.tensor(all_predictions)
        all_targets = torch.tensor(all_targets)
        all_probabilities = torch.tensor(all_probabilities)
        
        # Calculate metrics
        test_metrics = self.metrics_calculator.calculate_metrics(
            all_predictions, all_targets, all_probabilities
        )
        
        # Calculate calibration metrics
        calibration_metrics = self.calibration_calculator.calculate_calibration_metrics(
            all_probabilities, all_targets
        )
        test_metrics.update(calibration_metrics)
        
        # Log results
        self.logger.info("Test Results:")
        for metric, value in test_metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        # Save evaluation results
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            import json
            with open(save_dir / "test_metrics.json", 'w') as f:
                json.dump(test_metrics, f, indent=2)
            
            # Create visualizations
            visualizer = MetricsVisualizer(save_dir)
            
            # Plot confusion matrix
            visualizer.plot_confusion_matrix(
                all_predictions.numpy(),
                all_targets.numpy(),
                self.config['data']['class_names'],
                "Test Confusion Matrix",
                "confusion_matrix"
            )
            
            # Plot ROC curve
            if self.config['data']['num_classes'] == 2:
                visualizer.plot_roc_curve(
                    all_probabilities[:, 1].numpy(),
                    all_targets.numpy(),
                    "Test ROC Curve",
                    "roc_curve"
                )
                
                # Plot precision-recall curve
                visualizer.plot_precision_recall_curve(
                    all_probabilities[:, 1].numpy(),
                    all_targets.numpy(),
                    "Test Precision-Recall Curve",
                    "precision_recall_curve"
                )
                
                # Plot calibration curve
                visualizer.plot_calibration_curve(
                    all_probabilities[:, 1].numpy(),
                    all_targets.numpy(),
                    title="Test Calibration Curve",
                    save_name="calibration_curve"
                )
        
        return test_metrics
