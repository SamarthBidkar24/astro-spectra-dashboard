
import os
import sys
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_data
from src.preprocessing import get_feature_matrix, get_labels, encode_labels_multiclass, stratified_split
from src.models import create_conv_model

def train_convnet(core_path, model_dir, metrics_dir, plots_dir):
    print("Loading data...")
    asteroids_df = load_data(core_path)
    
    print("Preprocessing...")
    X = get_feature_matrix(asteroids_df)
    y = get_labels(asteroids_df)
    
    # Expand dims for ConvNet (samples, timesteps, features) -> (samples, 49, 1)
    X = np.expand_dims(X, axis=2)
    
    # Split
    X_train, X_test, y_train_orig, y_test_orig = stratified_split(X, y)
    
    # Encode
    label_encoder, y_train_enc = encode_labels_multiclass(y_train_orig)
    y_test_enc = label_encoder.transform(y_test_orig.reshape(-1, 1)).toarray()
    
    # Sample Weights
    sample_weight = compute_sample_weight("balanced", y=y_train_orig)
    
    # Create Model
    input_shape = (X_train.shape[1], 1)
    num_classes = y_train_enc.shape[1]
    
    model = create_conv_model(input_shape, n_outputs=num_classes)
    
    # Adapt Normalization
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Normalization):
            print("Adapting Normalization layer...")
            layer.adapt(X_train)
            break
            
    # Train
    print("Training ConvNet Model...")
    history = model.fit(
        X_train, y_train_enc,
        epochs=100,
        batch_size=32,
        validation_split=0.25,
        sample_weight=sample_weight,
        verbose=1
    )
    
    # Evaluate
    y_pred_prob = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_prob, axis=1)
    y_pred_labels = label_encoder.inverse_transform(y_pred_indices)
    
    f1 = f1_score(y_test_orig, y_pred_labels, average='weighted')
    cm = confusion_matrix(y_test_orig, y_pred_labels, labels=label_encoder.classes_)
    report = classification_report(y_test_orig, y_pred_labels)
    
    print(f"F1 Score: {f1}")
    
    # Save Model
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "conv_model"))
    
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder_conv.joblib"))
    print("Model and Encoder saved.")
    
    # Save Metrics
    os.makedirs(metrics_dir, exist_ok=True)
    metrics = {
        "f1_score": f1,
        "classification_report": report,
        "epochs": 100
    }
    with open(os.path.join(metrics_dir, "conv_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Plots
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('ConvNet Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "conv_loss.png"))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (ConvNet)')
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
    plt.yticks(tick_marks, label_encoder.classes_)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(plots_dir, "conv_confusion_matrix.png"))
    print("Plots saved.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    train_convnet(root_dir, 
              os.path.join(root_dir, "models"), 
              os.path.join(root_dir, "metrics"), 
              os.path.join(root_dir, "plots"))
