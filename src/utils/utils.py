"""Utility functions for diabetic retinopathy detection project."""

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Optional, List, Union
import logging
from pathlib import Path


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True


def get_device(fallback_order: List[str] = None) -> torch.device:
    """Get the best available device with fallback support.
    
    Args:
        fallback_order: List of device types to try in order
        
    Returns:
        Available torch device
    """
    if fallback_order is None:
        fallback_order = ["cuda", "mps", "cpu"]
    
    for device_type in fallback_order:
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_type == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif device_type == "cpu":
            return torch.device("cpu")
    
    return torch.device("cpu")


def setup_logging(log_dir: Union[str, Path], level: str = "INFO") -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("dr_detection")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: dict,
    filepath: Union[str, Path],
    is_best: bool = False
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'is_best': is_best
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: Union[str, Path]
) -> dict:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        filepath: Path to checkpoint file
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: PyTorch model
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
            
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model weights."""
        self.best_weights = model.state_dict().copy()
