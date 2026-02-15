
import os
import sys
import json
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import data_pipeline
from src import models_svm # Needed for SVMWrapper unpickling if class is not in __main__? 
# Actually, if SVMWrapper is defined in models_svm, unpickling might require models_svm to be imported.
# But joblib/pickle saves the module path. 'src.models_svm.SVMWrapper'. 
# Since we import src.models_svm, it should be fine.

def load_history_from_log(log_path='log.txt'):
    """
    Parses log.txt to extract training history for Dense and ConvNet models.
    Returns a dictionary: {'dense': {'loss': [], 'val_loss': [], ...}, 'convnet': ...}
    """
    history = {
        'dense': {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'epochs': []},
        'convnet': {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'epochs': []}
    }
    
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found. Cannot plot training history.")
        return history

    try:
        # Try finding encoding, default to utf-8 if utf-16 fails
        try:
            with open(log_path, 'r', encoding='utf-16') as f:
                content = f.read()
        except UnicodeError:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

        # Split content by model section
        # patterns in train_all_models.py:
        # "--- Training Dense Neural Network ---"
        # "--- Training 1D ConvNet ---"
        
        sections = re.split(r'--- Training (.*?) ---', content)
        
        current_model = None
        
        # Regex for Keras verbose=2 output
        # Example: "180/180 - 1s - loss: 1.0924 - accuracy: 0.6389 - val_loss: 0.8123 - val_accuracy: 0.7500"
        # Or verbose=1: "Epoch 1/100 ... loss: ... "
        
        # improved regex to catch floats
        pattern = r"loss:\s*([0-9\.]+).*?accuracy:\s*([0-9\.]+).*?val_loss:\s*([0-9\.]+).*?val_accuracy:\s*([0-9\.]+)"
        
        for i in range(len(sections)):
            header = sections[i-1] if i > 0 else ""
            stats = sections[i]
            
            if "Dense Neural Network" in header:
                current_model = 'dense'
            elif "1D ConvNet" in header:
                current_model = 'convnet'
            else:
                current_model = None
                
            if current_model:
                matches = re.findall(pattern, stats)
                for epoch_idx, match in enumerate(matches):
                    l, a, vl, va = match
                    history[current_model]['loss'].append(float(l))
                    history[current_model]['accuracy'].append(float(a))
                    history[current_model]['val_loss'].append(float(vl))
                    history[current_model]['val_accuracy'].append(float(va))
                    history[current_model]['epochs'].append(epoch_idx + 1)
                    
    except Exception as e:
        print(f"Error parsing log: {e}")

    return history

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    plt.figure(figsize=(10, 8))
    # classes might not be present in y_test, so we use the full class list for axis
    # But confusion_matrix(labels=classes) ensures shape
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Normalize? User didn't specify, but counts are standard.
    # User said "readable axis labels using class names".
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_history(hist_data, model_name, output_path):
    if not hist_data['epochs']:
        print(f"No history found for {model_name}, skipping history plot.")
        return

    epochs = hist_data['epochs']
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist_data['loss'], label='Train Loss')
    plt.plot(epochs, hist_data['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist_data['accuracy'], label='Train Acc')
    plt.plot(epochs, hist_data['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_metrics(y_true, y_pred, classes, output_path):
    # Overall
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class
    p_class, r_class, f1_class, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, average=None, zero_division=0)
    
    metrics = {
        "overall_accuracy": acc,
        "macro_metrics": {
            "precision": p_macro,
            "recall": r_macro,
            "f1": f1_macro
        },
        "per_class_metrics": {}
    }
    
    for i, cls in enumerate(classes):
        metrics["per_class_metrics"][cls] = {
            "precision": p_class[i],
            "recall": r_class[i],
            "f1": f1_class[i],
            "support": int(support[i])
        }
        
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved metrics to {output_path}")

def main():
    # Setup directories
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # 1. Load Data
    print("Loading Data...")
    # Use same split as train_all_models.py
    X_train, X_test, y_train, y_test = data_pipeline.get_train_test_split(test_size=0.2, random_state=42)
    
    # 2. Load History likely from log (since models were not saved with history)
    history_data = load_history_from_log('log.txt')
    
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    # Load Class Map
    class_map_path = os.path.join(models_dir, "class_map.json")
    if os.path.exists(class_map_path):
        with open(class_map_path, 'r') as f:
            class_names = json.load(f)
    else:
        print("Error: class_map.json not found!")
        sys.exit(1)
        
    # ==========================
    # SVM Evaluation
    # ==========================
    print("\n--- Evaluating SVM ---")
    svm_path = os.path.join(models_dir, "svm_multiclass.pkl")
    if os.path.exists(svm_path):
        svm_wrapper = joblib.load(svm_path)
        y_pred_svm = svm_wrapper.predict(X_test)
        
        # Save metrics
        save_metrics(y_test, y_pred_svm, class_names, "metrics/svm_metrics.json")
        # Plot Confusion
        plot_confusion_matrix(y_test, y_pred_svm, class_names, "plots/svm_confusion.png")
    else:
        print(f"SVM model not found at {svm_path}")
        
    # ==========================
    # Dense NN Evaluation
    # ==========================
    print("\n--- Evaluating Dense NN ---")
    dense_path = os.path.join(models_dir, "dense_multiclass.h5")
    if os.path.exists(dense_path):
        dense_model = keras.models.load_model(dense_path)
        
        # Predict
        y_pred_prob_dense = dense_model.predict(X_test)
        y_pred_idx_dense = np.argmax(y_pred_prob_dense, axis=1)
        y_pred_dense = np.array([class_names[i] for i in y_pred_idx_dense])
        
        # Save metrics
        save_metrics(y_test, y_pred_dense, class_names, "metrics/dense_metrics.json")
        # Plot Confusion
        plot_confusion_matrix(y_test, y_pred_dense, class_names, "plots/dense_confusion.png")
        
        # Plot History
        plot_history(history_data['dense'], "dense", "plots/dense_history.png")
    else:
        print(f"Dense model not found at {dense_path}")

    # ==========================
    # ConvNet Evaluation
    # ==========================
    print("\n--- Evaluating ConvNet ---")
    conv_path = os.path.join(models_dir, "convnet_multiclass.h5")
    if os.path.exists(conv_path):
        conv_model = keras.models.load_model(conv_path)
        
        # Prepare data: Expand Dims
        X_test_conv = np.expand_dims(X_test, axis=2)
        
        # Predict
        y_pred_prob_conv = conv_model.predict(X_test_conv)
        y_pred_idx_conv = np.argmax(y_pred_prob_conv, axis=1)
        y_pred_conv = np.array([class_names[i] for i in y_pred_idx_conv])
        
        # Save metrics
        save_metrics(y_test, y_pred_conv, class_names, "metrics/convnet_metrics.json")
        # Plot Confusion
        plot_confusion_matrix(y_test, y_pred_conv, class_names, "plots/convnet_confusion.png")
        
        # Plot History
        plot_history(history_data['convnet'], "convnet", "plots/convnet_history.png")
    else:
        print(f"ConvNet model not found at {conv_path}")

    print("\nEvaluation Completed.")

if __name__ == "__main__":
    main()
