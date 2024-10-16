#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

def evaluate_model(X_test, y_true, model, model_name):
    """
    Evaluate the model performance by generating a confusion matrix and ROC curve.
    
    Parameters:
    - X_test: The features used for testing.
    - y_true: The true labels (encoded as 0 and 1).
    - model: The trained model (can be a logistic regression, random forest, or neural network).
    - model_name: The name of the model (e.g., 'Logistic Regression', 'Random Forest', 'Neural Network').
    """
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(y_true)
    # Check if the model has `predict_proba` for probability-based predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
    else:
        # If it's a neural network, use the .predict() method to get probabilities
        y_pred_proba = model.predict(X_test).ravel()  # Neural networks often output probabilities directly

    # Convert predicted probabilities to binary for confusion matrix
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred_binary, model_name)
    
    # ROC Curve
    plot_roc_curve(y_true, y_pred_proba, model_name)

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plots and saves the confusion matrix as a .jpg image.
    
    Parameters:
    - y_true: The true binary labels.
    - y_pred: The predicted binary labels.
    - model_name: The name of the model for the plot title.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Attrited Customer", "Existing Customer"],
                yticklabels=["Attrited Customer", "Existing Customer"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    if not os.path.isfile(f"confusion_matrix_{model_name}.jpg"):
        plt.savefig(f"./figures/confusion_matrix_{model_name}.jpg")
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, model_name):
    """
    Plots and saves the ROC curve as a .jpg image.
    
    Parameters:
    - y_true: The true binary labels.
    - y_pred_proba: The predicted probabilities.
    - model_name: The name of the model for the plot title.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    if not os.path.isfile(f"roc_curve_{model_name}.jpg"):
        plt.savefig(f"./figures/roc_curve_{model_name}.jpg")
    plt.show()
