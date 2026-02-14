
import os
import sys
import argparse
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_data
from src.preprocessing import get_feature_matrix, get_labels, scale_data, stratified_split
from src.models import get_svm_model

def train_svm(core_path, model_dir, metrics_dir, plots_dir):
    print("Loading data...")
    asteroids_df = load_data(core_path)
    
    print("Preprocessing...")
    X = get_feature_matrix(asteroids_df)
    y = get_labels(asteroids_df)
    
    # Split
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Scale
    scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # Class weights
    weight_dict = {}
    classes = np.unique(y_train)
    for ast_type in classes:
        weight_dict[ast_type] = float(1.0 / (len(y_train[y_train == ast_type]) / (len(y_train))))
    print(f"Class weights: {weight_dict}")
    
    # Grid Search
    print("Starting Grid Search...")
    param_grid = [
      {'C': np.logspace(0, 3.5, 10), 'kernel': ['linear']},
      {'C': np.logspace(0, 3.5, 10), 'kernel': ['rbf']},
    ]
    
    svc = get_svm_model(class_weight=weight_dict)
    
    # Using smaller cv for speed in this context
    clf = GridSearchCV(svc, param_grid, scoring='f1_weighted', verbose=1, cv=3, n_jobs=1)
    clf.fit(X_train_scaled, y_train)
    
    best_model = clf.best_estimator_
    print(f"Best parameters: {clf.best_params_}")
    
    # Evaluate
    print("Evaluating...")
    y_pred = best_model.predict(X_test_scaled)
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report = classification_report(y_test, y_pred)
    
    print(f"F1 Score: {f1}")
    print(report)
    
    # Save Model and Scaler
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "svm_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    
    # Save Metrics
    os.makedirs(metrics_dir, exist_ok=True)
    metrics = {
        "f1_score": f1,
        "best_params": clf.best_params_,
        "classification_report": report
    }
    with open(os.path.join(metrics_dir, "svm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Confusion Matrix Plot
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (SVM)')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(plots_dir, "svm_confusion_matrix.png"))
    print("Plots saved.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    train_svm(root_dir, 
              os.path.join(root_dir, "models"), 
              os.path.join(root_dir, "metrics"), 
              os.path.join(root_dir, "plots"))
