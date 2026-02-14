
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import data_pipeline
from src import models_dense

def main():
    print("Debug: Loading data...")
    data_pipeline.build_clean_dataset()
    X_train, X_test, y_train, y_test = data_pipeline.get_train_test_split(test_size=0.2, random_state=42)
    
    # Split for DL
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for t_index, v_index in sss.split(X_train, y_train):
        X_train_dl, X_val_dl = X_train[t_index], X_train[v_index]
        y_train_dl, y_val_dl = y_train[t_index], y_train[v_index]
        
    print("Debug: Training Dense (1 epoch)...")
    # Train 1 epoch
    model, hist = models_dense.train_model(X_train_dl, y_train_dl, X_val_dl, y_val_dl, epochs=1, batch_size=32)
    
    print("Debug: Dense Trained.")
    models_dense.evaluate_model(model, X_test, y_test)
    print("Debug: Done.")

if __name__ == "__main__":
    main()
