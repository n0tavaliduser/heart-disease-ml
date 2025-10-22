"""
Model evaluation module for the heart disease prediction project.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from pathlib import Path
from .utils import RESULTS_DIR

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a single model using various metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all models and return their metrics.
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics for all models
    """
    all_metrics = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        all_metrics[name] = evaluate_model(model, X_test, y_test)
    return all_metrics

def plot_roc_curves(models, X_test, y_test):
    """
    Plot ROC curves for all models.
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(Path(RESULTS_DIR) / 'plot' / 'roc_curves.png')
    plt.close()

def plot_confusion_matrices(models, X_test, y_test):
    """
    Plot confusion matrices for all models.
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
    """
    n_models = len(models)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name}\nConfusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Remove empty subplots if any
    for ax in axes[len(models):]:
        ax.remove()
    
    plt.tight_layout()
    plt.savefig(Path(RESULTS_DIR) / 'plot' / 'confusion_matrices.png')
    plt.close()

def plot_feature_importance(feature_importance, title):
    """
    Plot feature importance.
    
    Args:
        feature_importance (pd.Series): Feature importance scores
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title(f'Feature Importance - {title}')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(Path(RESULTS_DIR) / 'plot' / f'feature_importance_{title.lower()}.png')
    plt.close()
