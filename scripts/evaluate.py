#!/usr/bin/env python3
"""Evaluation script for diabetic retinopathy detection models."""

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
    parser = argparse.ArgumentParser(description="Evaluate diabetic retinopathy detection model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    config.seed = args.seed
    
    # Set random seed
    set_seed(config.seed, config.deterministic)
    
    # Setup device
    device = get_device(config.device.fallback_order)
    print(f"Using device: {device}")
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir / "logs", config.logging.level)
    logger.info("Starting diabetic retinopathy detection evaluation")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Create trainer for evaluation
    trainer = Trainer(model, config, device, logger)
    
    # Run evaluation
    logger.info("Running evaluation...")
    test_metrics = trainer.evaluate(test_loader, output_dir)
    
    # Log results
    logger.info("Evaluation completed")
    logger.info("Test Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    import json
    results = {
        'checkpoint': str(args.checkpoint),
        'config': dict(config),
        'metrics': test_metrics
    }
    
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
