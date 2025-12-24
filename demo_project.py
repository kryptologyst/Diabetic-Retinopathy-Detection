#!/usr/bin/env python3
"""Demo script to showcase the diabetic retinopathy detection project."""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.utils import set_seed, get_device
from src.data.dataset import SyntheticRetinaDataset, get_transforms
from src.models.models import create_model
from src.losses.losses import create_loss_function
from src.metrics.metrics import MetricsCalculator


def main():
    """Demonstrate the diabetic retinopathy detection system."""
    print("🔬 Diabetic Retinopathy Detection - Research Demo")
    print("=" * 60)
    
    # Set up
    set_seed(42, deterministic=True)
    device = get_device()
    print(f"🖥️  Using device: {device}")
    
    # Configuration
    config = {
        'model': {
            'name': 'efficientnet_b0',
            'num_classes': 2,
            'pretrained': True,
            'dropout': 0.2
        },
        'data': {
            'num_classes': 2,
            'class_names': ['No DR', 'DR'],
            'input_size': [224, 224]
        },
        'loss': {
            'name': 'focal',
            'focal_alpha': 0.25,
            'focal_gamma': 2.0
        }
    }
    
    print("\n📊 1. Creating Synthetic Dataset")
    print("-" * 40)
    
    # Create synthetic dataset
    dataset = SyntheticRetinaDataset(size=100, image_size=(224, 224))
    print(f"✅ Created dataset with {len(dataset)} synthetic retinal images")
    
    # Show class distribution
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    print(f"📈 Class distribution: No DR={class_counts[0]}, DR={class_counts[1]}")
    
    print("\n🧠 2. Creating Model")
    print("-" * 40)
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Created {config['model']['name']} model")
    print(f"📊 Model parameters: {param_count:,}")
    
    print("\n🎯 3. Testing Model Inference")
    print("-" * 40)
    
    # Test inference
    model.eval()
    with torch.no_grad():
        # Get a sample image
        image, true_label = dataset[0]
        
        # Convert to tensor
        transform = get_transforms(input_size=(224, 224), is_training=False)
        transformed = transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Make prediction
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities.max().item()
        
        print(f"🔍 Sample prediction:")
        print(f"   True label: {config['data']['class_names'][true_label]}")
        print(f"   Predicted: {config['data']['class_names'][prediction]}")
        print(f"   Confidence: {confidence:.1%}")
        
        # Show probabilities
        probs = probabilities[0].cpu().numpy()
        for i, (class_name, prob) in enumerate(zip(config['data']['class_names'], probs)):
            print(f"   {class_name}: {prob:.1%}")
    
    print("\n📈 4. Evaluating Model Performance")
    print("-" * 40)
    
    # Create test dataset
    test_dataset = SyntheticRetinaDataset(size=50, image_size=(224, 224))
    
    # Evaluate on test set
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, target = test_dataset[i]
            
            # Transform image
            transformed = transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0).to(device)
            
            # Make prediction
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            
            all_predictions.append(prediction)
            all_targets.append(target)
            all_probabilities.append(probabilities[0].cpu().numpy())
    
    # Calculate metrics
    calculator = MetricsCalculator(num_classes=2, class_names=config['data']['class_names'])
    metrics = calculator.calculate_metrics(
        torch.tensor(all_predictions),
        torch.tensor(all_targets),
        torch.tensor(all_probabilities)
    )
    
    print("📊 Performance Metrics:")
    print(f"   Accuracy: {metrics['accuracy']:.1%}")
    if 'auroc' in metrics:
        print(f"   AUROC: {metrics['auroc']:.3f}")
    if 'sensitivity' in metrics:
        print(f"   Sensitivity: {metrics['sensitivity']:.1%}")
    if 'specificity' in metrics:
        print(f"   Specificity: {metrics['specificity']:.1%}")
    
    print("\n🎨 5. Visualizing Sample Images")
    print("-" * 40)
    
    # Show sample images
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(6):
        image, label = test_dataset[i]
        axes[i].imshow(image)
        axes[i].set_title(f"{config['data']['class_names'][label]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    print("✅ Saved sample images to 'sample_images.png'")
    
    print("\n🔧 6. Testing Loss Functions")
    print("-" * 40)
    
    # Test loss function
    loss_fn = create_loss_function(config)
    
    # Create dummy data
    logits = torch.randn(10, 2)
    targets = torch.randint(0, 2, (10,))
    
    loss = loss_fn(logits, targets)
    print(f"✅ Focal loss: {loss.item():.4f}")
    
    print("\n🎉 Demo Complete!")
    print("=" * 60)
    print("🚀 Next Steps:")
    print("   1. Run training: python scripts/train.py")
    print("   2. Start demo: streamlit run demo/app.py")
    print("   3. Run tests: pytest tests/ -v")
    print("\n⚠️  Remember: This is for research purposes only!")
    print("   Not intended for clinical diagnosis or medical decisions.")


if __name__ == "__main__":
    main()
