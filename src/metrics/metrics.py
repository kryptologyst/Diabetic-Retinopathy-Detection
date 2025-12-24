"""Evaluation metrics for diabetic retinopathy detection."""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report
)
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        """Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth labels
            probabilities: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        # Basic classification metrics
        metrics.update(self._calculate_classification_metrics(predictions, targets))
        
        # Probability-based metrics
        if probabilities is not None:
            metrics.update(self._calculate_probability_metrics(probabilities, targets))
        
        # Confusion matrix metrics
        metrics.update(self._calculate_confusion_matrix_metrics(predictions, targets))
        
        return metrics
    
    def _calculate_classification_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = np.mean(predictions == targets)
        
        # Per-class metrics
        for i in range(self.num_classes):
            class_mask = targets == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(predictions[class_mask] == i)
                metrics[f'accuracy_class_{i}'] = class_accuracy
        
        return metrics
    
    def _calculate_probability_metrics(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate probability-based metrics."""
        metrics = {}
        
        if self.num_classes == 2:
            # Binary classification
            if probabilities.ndim > 1:
                prob_positive = probabilities[:, 1]
            else:
                prob_positive = probabilities
            
            # AUC-ROC
            try:
                metrics['auroc'] = roc_auc_score(targets, prob_positive)
            except ValueError:
                metrics['auroc'] = 0.0
            
            # AUC-PR
            try:
                metrics['auprc'] = average_precision_score(targets, prob_positive)
            except ValueError:
                metrics['auprc'] = 0.0
            
            # Sensitivity and Specificity
            metrics.update(self._calculate_sensitivity_specificity(prob_positive, targets))
            
        else:
            # Multi-class classification
            # One-vs-rest AUC
            for i in range(self.num_classes):
                binary_targets = (targets == i).astype(int)
                if np.sum(binary_targets) > 0:
                    try:
                        metrics[f'auroc_class_{i}'] = roc_auc_score(
                            binary_targets, probabilities[:, i]
                        )
                    except ValueError:
                        metrics[f'auroc_class_{i}'] = 0.0
        
        return metrics
    
    def _calculate_sensitivity_specificity(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate sensitivity and specificity at optimal threshold."""
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in thresholds:
            predictions_thresh = (probabilities >= threshold).astype(int)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(targets, predictions_thresh).ravel()
            
            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = sensitivity
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Calculate final metrics at best threshold
        predictions_best = (probabilities >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(targets, predictions_best).ravel()
        
        metrics = {
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'f1_score': best_f1,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0.0,  # Positive Predictive Value
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0,  # Negative Predictive Value
            'optimal_threshold': best_threshold
        }
        
        return metrics
    
    def _calculate_confusion_matrix_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate confusion matrix based metrics."""
        metrics = {}
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Per-class precision, recall, F1
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[f'precision_class_{i}'] = precision
            metrics[f'recall_class_{i}'] = recall
            metrics[f'f1_class_{i}'] = f1
        
        return metrics


class CalibrationCalculator:
    """Calculator for model calibration metrics."""
    
    def __init__(self, n_bins: int = 10):
        """Initialize calibration calculator.
        
        Args:
            n_bins: Number of bins for calibration
        """
        self.n_bins = n_bins
    
    def calculate_calibration_metrics(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate calibration metrics.
        
        Args:
            probabilities: Predicted probabilities
            targets: Ground truth labels
            
        Returns:
            Dictionary of calibration metrics
        """
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        if probabilities.ndim > 1:
            probabilities = probabilities[:, 1]  # Use positive class probability
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(probabilities, targets)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(probabilities, targets)
        
        # Brier Score
        brier_score = self._calculate_brier_score(probabilities, targets)
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score
        }
    
    def _calculate_ece(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _calculate_brier_score(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Calculate Brier Score."""
        return np.mean((probabilities - targets) ** 2)


class MetricsVisualizer:
    """Visualizer for evaluation metrics."""
    
    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        """Initialize metrics visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        save_name: Optional[str] = None
    ) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        title: str = "ROC Curve",
        save_name: Optional[str] = None
    ) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auc = roc_auc_score(targets, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        title: str = "Precision-Recall Curve",
        save_name: Optional[str] = None
    ) -> None:
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        auc_pr = average_precision_score(targets, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {auc_pr:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10,
        title: str = "Calibration Curve",
        save_name: Optional[str] = None
    ) -> None:
        """Plot calibration curve."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(targets[in_bin].mean())
                bin_confidences.append(probabilities[in_bin].mean())
        
        plt.figure(figsize=(8, 6))
        plt.plot(bin_confidences, bin_accuracies, 'o-', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.show()


def create_metrics_calculator(config: dict) -> MetricsCalculator:
    """Create metrics calculator from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Metrics calculator
    """
    return MetricsCalculator(
        num_classes=config['data']['num_classes'],
        class_names=config['data']['class_names']
    )
