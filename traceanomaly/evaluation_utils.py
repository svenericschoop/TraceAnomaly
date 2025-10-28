"""
Evaluation utilities for TraceAnomaly testing.

This module provides functions for calculating metrics, applying KDE-based thresholding,
and generating evaluation reports as described in the TraceAnomaly paper.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
import json
from typing import Dict, List, Tuple, Any


def calculate_kde_threshold(normal_scores: np.ndarray, significance_level: float = 0.001) -> float:
    """
    Calculate threshold using KDE-based approach as described in paper Section III-D.
    
    Args:
        normal_scores: Array of log-likelihood scores from normal traces
        significance_level: P-value threshold for anomaly detection (default: 0.001)
        
    Returns:
        Threshold value for anomaly detection
    """
    if len(normal_scores) == 0:
        raise ValueError("No normal scores provided for KDE fitting")
    
    # Fit KDE on normal scores
    kde = gaussian_kde(normal_scores)
    
    # Find threshold where p-value = significance_level
    # We need to find the score where the cumulative probability is (1 - significance_level)
    sorted_scores = np.sort(normal_scores)
    
    # Use percentile-based approach for threshold
    threshold_percentile = (1 - significance_level) * 100
    threshold = np.percentile(sorted_scores, threshold_percentile)
    
    return threshold


def apply_kde_thresholding(test_scores: np.ndarray, normal_scores: np.ndarray, 
                         significance_level: float = 0.001) -> np.ndarray:
    """
    Apply KDE-based thresholding to classify test scores as normal/anomalous.
    
    Args:
        test_scores: Array of log-likelihood scores from test traces
        normal_scores: Array of log-likelihood scores from normal training traces
        significance_level: P-value threshold for anomaly detection
        
    Returns:
        Boolean array where True indicates anomaly
    """
    if len(normal_scores) == 0:
        raise ValueError("No normal scores provided for KDE fitting")
    
    # Fit KDE on normal scores
    kde = gaussian_kde(normal_scores)
    
    # Calculate p-values for test scores
    # P-value is the probability of observing a score as extreme or more extreme
    # For anomaly detection, we want low p-values (scores that are unlikely under normal distribution)
    p_values = []
    for score in test_scores:
        # Calculate the probability of observing a score <= this score
        # This gives us the p-value for the left tail
        p_value = kde.integrate_box_1d(-np.inf, score)
        p_values.append(p_value)
    
    p_values = np.array(p_values)
    
    # Classify as anomaly if p-value < significance_level
    is_anomaly = p_values < significance_level
    
    return is_anomaly


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels (0=normal, 1=anomalous)
        y_pred: Predicted labels (0=normal, 1=anomalous)
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Calculate additional metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0
    })
    
    return metrics


def generate_confusion_matrix_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate confusion matrix as formatted text.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Formatted confusion matrix string
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create formatted confusion matrix
    text = "Confusion Matrix:\n"
    text += "=" * 30 + "\n"
    text += f"{'':>10} {'Predicted':>15}\n"
    text += f"{'':>10} {'Normal':>7} {'Anomalous':>8}\n"
    text += f"{'Actual':>10} {'Normal':>7} {cm[0,0]:>8} {cm[0,1]:>8}\n"
    text += f"{'':>10} {'Anomalous':>7} {cm[1,0]:>8} {cm[1,1]:>8}\n"
    text += "=" * 30 + "\n"
    
    return text


def calculate_score_statistics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Calculate statistics for anomaly scores by label.
    
    Args:
        scores: Array of anomaly scores
        labels: Array of ground truth labels
        
    Returns:
        Dictionary containing score statistics
    """
    normal_scores = scores[labels == 0]
    anomalous_scores = scores[labels == 1]
    
    stats = {
        'normal_scores': {
            'count': len(normal_scores),
            'mean': float(np.mean(normal_scores)) if len(normal_scores) > 0 else 0.0,
            'std': float(np.std(normal_scores)) if len(normal_scores) > 0 else 0.0,
            'min': float(np.min(normal_scores)) if len(normal_scores) > 0 else 0.0,
            'max': float(np.max(normal_scores)) if len(normal_scores) > 0 else 0.0,
            'median': float(np.median(normal_scores)) if len(normal_scores) > 0 else 0.0
        },
        'anomalous_scores': {
            'count': len(anomalous_scores),
            'mean': float(np.mean(anomalous_scores)) if len(anomalous_scores) > 0 else 0.0,
            'std': float(np.std(anomalous_scores)) if len(anomalous_scores) > 0 else 0.0,
            'min': float(np.min(anomalous_scores)) if len(anomalous_scores) > 0 else 0.0,
            'max': float(np.max(anomalous_scores)) if len(anomalous_scores) > 0 else 0.0,
            'median': float(np.median(anomalous_scores)) if len(anomalous_scores) > 0 else 0.0
        }
    }
    
    return stats


def save_evaluation_results(results_df: pd.DataFrame, metrics: Dict[str, float], 
                          output_file: str, confusion_matrix_text: str = None) -> None:
    """
    Save evaluation results to files.
    
    Args:
        results_df: DataFrame with detailed results per trace
        metrics: Dictionary of calculated metrics
        output_file: Path to main results CSV file
        confusion_matrix_text: Formatted confusion matrix text
    """
    # Save detailed results
    results_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    
    # Save metrics summary
    metrics_file = output_file.replace('.csv', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics summary saved to: {metrics_file}")
    
    # Save confusion matrix
    if confusion_matrix_text:
        cm_file = output_file.replace('.csv', '_confusion_matrix.txt')
        with open(cm_file, 'w') as f:
            f.write(confusion_matrix_text)
        print(f"Confusion matrix saved to: {cm_file}")


def print_evaluation_summary(metrics: Dict[str, float], score_stats: Dict[str, Any]) -> None:
    """
    Print a formatted summary of evaluation results.
    
    Args:
        metrics: Dictionary of calculated metrics
        score_stats: Dictionary of score statistics
    """
    print("\n" + "=" * 60)
    print("TRACEANOMALY EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    print(f"\nCONFUSION MATRIX:")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Positives:  {metrics['true_positives']}")
    
    print(f"\nSCORE STATISTICS:")
    print(f"  Normal traces ({score_stats['normal_scores']['count']}):")
    print(f"    Mean: {score_stats['normal_scores']['mean']:.4f}")
    print(f"    Std:  {score_stats['normal_scores']['std']:.4f}")
    print(f"    Range: [{score_stats['normal_scores']['min']:.4f}, {score_stats['normal_scores']['max']:.4f}]")
    
    print(f"  Anomalous traces ({score_stats['anomalous_scores']['count']}):")
    print(f"    Mean: {score_stats['anomalous_scores']['mean']:.4f}")
    print(f"    Std:  {score_stats['anomalous_scores']['std']:.4f}")
    print(f"    Range: [{score_stats['anomalous_scores']['min']:.4f}, {score_stats['anomalous_scores']['max']:.4f}]")
    
    print("=" * 60)


def validate_evaluation_inputs(test_scores: np.ndarray, test_labels: np.ndarray, 
                              normal_scores: np.ndarray) -> None:
    """
    Validate inputs for evaluation functions.
    
    Args:
        test_scores: Array of test scores
        test_labels: Array of test labels
        normal_scores: Array of normal training scores
    """
    if len(test_scores) != len(test_labels):
        raise ValueError("Test scores and labels must have the same length")
    
    if len(test_scores) == 0:
        raise ValueError("No test data provided")
    
    if len(normal_scores) == 0:
        raise ValueError("No normal training scores provided for KDE fitting")
    
    unique_labels = np.unique(test_labels)
    if not np.array_equal(unique_labels, [0, 1]):
        raise ValueError("Test labels must contain only 0 (normal) and 1 (anomalous) values")
    
    if np.any(np.isnan(test_scores)) or np.any(np.isinf(test_scores)):
        raise ValueError("Test scores contain NaN or infinite values")
    
    if np.any(np.isnan(normal_scores)) or np.any(np.isinf(normal_scores)):
        raise ValueError("Normal scores contain NaN or infinite values")
