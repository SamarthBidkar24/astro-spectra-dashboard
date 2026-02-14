
import os
import sys
import json
import joblib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import data_pipeline
from src import models_svm
from src import models_dense
from src import models_convnet

def main():
    print("Step 1: Building/Loading Clean Dataset...")
    data_pipeline.build_clean_dataset()
    
    print("\nStep 2: Splitting Data...")
    X_train, X_test, y_train, y_test = data_pipeline.get_train_test_split(test_size=0.2, random_state=42)
    print(f"Total Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Create Validation Split for DL models (25% of Train = 20% of Total, leaving 60% for Training)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for t_index, v_index in sss.split(X_train, y_train):
        X_train_dl, X_val_dl = X_train[t_index], X_train[v_index]
        y_train_dl, y_val_dl = y_train[t_index], y_train[v_index]
    
    print(f"DL Train: {X_train_dl.shape}, DL Val: {X_val_dl.shape}")
    
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. SVM Multiclass
    # ---------------------------------------------------------
    print("\n--- Training Multiclass SVM ---")
    # SVM module handles scaling internally
    svm_wrapper = models_svm.train_model(X_train, y_train)
    
    print("Evaluating SVM...")
    # SVM module evaluation
    models_svm.evaluate_model(svm_wrapper, X_test, y_test)
    
    # Save Model
    svm_path = os.path.join(models_dir, "svm_multiclass.pkl")
    joblib.dump(svm_wrapper, svm_path)
    print(f"SVM model saved to {svm_path}")
    
    # Save Scaler (extracted from wrapper)
    if hasattr(svm_wrapper, 'scaler'):
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        joblib.dump(svm_wrapper.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
    # ---------------------------------------------------------
    # 2. Dense Neural Network
    # ---------------------------------------------------------
    print("\n--- Training Dense Neural Network ---")
    dense_model, dense_hist = models_dense.train_model(X_train_dl, y_train_dl, X_val_dl, y_val_dl)
    
    print("Evaluating Dense NN...")
    models_dense.evaluate_model(dense_model, X_test, y_test)
    
    # Save Model
    dense_path = os.path.join(models_dir, "dense_multiclass.h5")
    dense_model.save(dense_path)
    print(f"Dense model saved to {dense_path}")
    
    # Save Class Map (from label encoder)
    if hasattr(dense_model, 'label_encoder'):
        # OneHotEncoder categories_ is a list of arrays (one per feature). We updated y to be 1D so index 0.
        classes = dense_model.label_encoder.categories_[0]
        class_map_path = os.path.join(models_dir, "class_map.json")
        with open(class_map_path, "w") as f:
            json.dump(classes.tolist(), f)
        print(f"Class map saved to {class_map_path}")
        
    # ---------------------------------------------------------
    # 3. 1D ConvNet
    # ---------------------------------------------------------
    print("\n--- Training 1D ConvNet ---")
    # Expand dims for ConvNet: (N, 49) -> (N, 49, 1)
    X_train_conv = np.expand_dims(X_train_dl, axis=2)
    X_val_conv = np.expand_dims(X_val_dl, axis=2)
    X_test_conv = np.expand_dims(X_test, axis=2)
    
    conv_model, conv_hist = models_convnet.train_model(X_train_conv, y_train_dl, X_val_conv, y_val_dl)
    
    print("Evaluating ConvNet...")
    models_convnet.evaluate_model(conv_model, X_test_conv, y_test)
    
    # Save Model
    conv_path = os.path.join(models_dir, "convnet_multiclass.h5")
    conv_model.save(conv_path)
    print(f"ConvNet model saved to {conv_path}")

    print("\nAll models trained and saved successfully.")

if __name__ == "__main__":
    main()
