#!/usr/bin/env python3
"""Main training script for diabetic retinopathy detection."""

import argparse
import yaml
import torch
import logging
from pathlib import Path
from omegaconf import OmegaConf

from src.utils.utils import set_seed, get_device, setup_logging
from src.data.dataset import create_data_loaders
from src.models.models import create_model
from src.train.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train diabetic retinopathy detection model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate the model"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    config.seed = args.seed
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    
    # Set random seed
    set_seed(config.seed, config.deterministic)
    
    # Setup device
    device = get_device(config.device.fallback_order)
    print(f"Using device: {device}")
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir / "logs", config.logging.level)
    logger.info("Starting diabetic retinopathy detection training")
    logger.info(f"Configuration: {config}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, config, device, logger)
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
    
    if args.eval_only:
        # Evaluation only
        logger.info("Running evaluation only...")
        test_metrics = trainer.evaluate(test_loader, output_dir)
        
        logger.info("Evaluation completed")
        logger.info("Test Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    else:
        # Training
        logger.info("Starting training...")
        training_history = trainer.train(train_loader, val_loader, output_dir)
        
        # Final evaluation
        logger.info("Running final evaluation...")
        test_metrics = trainer.evaluate(test_loader, output_dir)
        
        logger.info("Training completed")
        logger.info("Final Test Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save training history
        import json
        with open(output_dir / "training_history.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_serializable = {}
            for key, values in training_history.items():
                if isinstance(values, list) and len(values) > 0:
                    if isinstance(values[0], dict):
                        # Metrics dictionaries
                        history_serializable[key] = values
                    else:
                        # Loss values
                        history_serializable[key] = [float(v) for v in values]
                else:
                    history_serializable[key] = values
            
            json.dump(history_serializable, f, indent=2)
        
        logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
