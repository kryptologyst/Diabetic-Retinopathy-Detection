"""Unit tests for diabetic retinopathy detection project."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.utils.utils import set_seed, get_device, count_parameters, EarlyStopping
from src.data.dataset import SyntheticRetinaDataset, get_transforms
from src.models.models import DRClassifier, AttentionDRClassifier, create_model
from src.losses.losses import FocalLoss, create_loss_function
from src.metrics.metrics import MetricsCalculator, CalibrationCalculator


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting functionality."""
        set_seed(42)
        
        # Test numpy random
        np.random.seed(42)
        val1 = np.random.random()
        np.random.seed(42)
        val2 = np.random.random()
        assert val1 == val2
        
        # Test torch random
        torch.manual_seed(42)
        val1 = torch.rand(1).item()
        torch.manual_seed(42)
        val2 = torch.rand(1).item()
        assert val1 == val2
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        
        # Test fallback order
        device = get_device(["cpu"])
        assert device.type == "cpu"
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        param_count = count_parameters(model)
        assert param_count == 55  # 10*5 + 5 (bias)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        model = torch.nn.Linear(10, 1)
        early_stopping = EarlyStopping(patience=3)
        
        # Test normal operation
        assert not early_stopping(0.5, model)
        assert not early_stopping(0.4, model)  # Improvement
        assert not early_stopping(0.5, model)  # No improvement
        assert not early_stopping(0.6, model)  # Worse
        assert early_stopping(0.7, model)  # Should stop


class TestDataset:
    """Test dataset functionality."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticRetinaDataset(size=10, image_size=(64, 64))
        
        assert len(dataset) == 10
        
        # Test data loading
        image, label = dataset[0]
        assert isinstance(image, np.ndarray)
        assert image.shape == (64, 64, 3)
        assert isinstance(label, int)
        assert label in [0, 1]
    
    def test_transforms(self):
        """Test data transforms."""
        transform = get_transforms(input_size=(64, 64), is_training=False)
        
        # Create dummy image
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Apply transform
        transformed = transform(image=image)
        assert 'image' in transformed
        assert transformed['image'].shape == (3, 64, 64)


class TestModels:
    """Test model architectures."""
    
    def test_dr_classifier(self):
        """Test basic DR classifier."""
        model = DRClassifier(
            model_name="resnet18",
            num_classes=2,
            pretrained=False
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (2, 2)
        assert not torch.isnan(output).any()
    
    def test_attention_classifier(self):
        """Test attention-based classifier."""
        model = AttentionDRClassifier(
            backbone_name="efficientnet_b0",
            num_classes=2,
            pretrained=False
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        logits, attention_map = model(x)
        
        assert logits.shape == (2, 2)
        assert attention_map.shape == (2, 1, 7, 7)  # EfficientNet-B0 feature map size
        assert not torch.isnan(logits).any()
        assert not torch.isnan(attention_map).any()
    
    def test_create_model(self):
        """Test model creation from config."""
        config = {
            'model': {
                'name': 'resnet18',
                'num_classes': 2,
                'pretrained': False,
                'dropout': 0.2
            },
            'data': {
                'num_classes': 2
            }
        }
        
        model = create_model(config)
        assert isinstance(model, DRClassifier)
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 2)


class TestLosses:
    """Test loss functions."""
    
    def test_focal_loss(self):
        """Test focal loss calculation."""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Create dummy data
        logits = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        
        loss = focal_loss(logits, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_create_loss_function(self):
        """Test loss function creation from config."""
        config = {
            'loss': {
                'name': 'focal',
                'focal_alpha': 0.25,
                'focal_gamma': 2.0
            },
            'data': {
                'num_classes': 2
            }
        }
        
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, FocalLoss)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_metrics_calculator(self):
        """Test metrics calculation."""
        calculator = MetricsCalculator(num_classes=2)
        
        # Create dummy predictions
        predictions = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 1, 0])
        probabilities = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
        
        metrics = calculator.calculate_metrics(predictions, targets, probabilities)
        
        assert 'accuracy' in metrics
        assert 'auroc' in metrics
        assert 'sensitivity' in metrics
        assert 'specificity' in metrics
    
    def test_calibration_calculator(self):
        """Test calibration metrics calculation."""
        calculator = CalibrationCalculator()
        
        # Create dummy probabilities and targets
        probabilities = torch.tensor([0.1, 0.3, 0.7, 0.9])
        targets = torch.tensor([0, 0, 1, 1])
        
        metrics = calculator.calculate_calibration_metrics(probabilities, targets)
        
        assert 'ece' in metrics
        assert 'mce' in metrics
        assert 'brier_score' in metrics


class TestIntegration:
    """Integration tests."""
    
    def test_training_pipeline(self):
        """Test complete training pipeline."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create small dataset
            dataset = SyntheticRetinaDataset(size=20, image_size=(64, 64))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
            
            # Create model
            model = DRClassifier(
                model_name="resnet18",
                num_classes=2,
                pretrained=False
            )
            
            # Create optimizer and loss
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Train for one epoch
            model.train()
            total_loss = 0
            
            for images, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            assert total_loss > 0
    
    def test_evaluation_pipeline(self):
        """Test evaluation pipeline."""
        # Create model
        model = DRClassifier(
            model_name="resnet18",
            num_classes=2,
            pretrained=False
        )
        
        # Create test data
        dataset = SyntheticRetinaDataset(size=10, image_size=(64, 64))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        # Evaluate
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in dataloader:
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.numpy())
                all_targets.extend(targets.numpy())
        
        # Calculate metrics
        calculator = MetricsCalculator(num_classes=2)
        metrics = calculator.calculate_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_targets)
        )
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
