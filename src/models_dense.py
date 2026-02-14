
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_sample_weight

def build_model(input_shape, n_outputs):
    """
    recreates the exact architecture used in the original notebook for that model type.
    Notebook 8: Dense network with Normalization layer.
    """
    # Note: Normalization layer needs .adapt(). This usually happens before compilation or fitting.
    # In notebook 8, it happens inside create_model before constructing the rest.
    # But adapt needs data X_train.
    # If build_model doesn't take X_train, we return a model with an un-adapted Normalization layer?
    # Or we handle adaption in train_model.
    # We will define structure here.
    
    input_layer = layers.Input(shape=input_shape)
    
    # Normalization layer
    normalizer = layers.Normalization(axis=1)
    # We cannot adapt here without data.
    
    hidden_layer = normalizer(input_layer)
    hidden_layer = layers.Dense(25)(hidden_layer)
    hidden_layer = layers.ReLU()(hidden_layer)
    
    hidden_layer = layers.Dense(10)(hidden_layer)
    hidden_layer = layers.ReLU()(hidden_layer)
    
    output_layer = layers.Dense(n_outputs, activation="softmax")(hidden_layer)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    trains and returns the model (and history for DL models).
    """
    # Prepare labels (OneHot)
    # Notebook 8 uses OneHotEncoder.
    # We assume y_train is raw class names (e.g. 'C', 'S'...).
    
    label_encoder = preprocessing.OneHotEncoder(sparse_output=False)
    y_train_enc = label_encoder.fit_transform(y_train.reshape(-1, 1))
    y_val_enc = label_encoder.transform(y_val.reshape(-1, 1))
    
    # Sample Weights
    sample_weight = compute_sample_weight("balanced", y=y_train)
    
    # Build
    input_shape = (X_train.shape[1],)
    num_classes = y_train_enc.shape[1]
    
    model = build_model(input_shape, num_classes)
    
    # Adapt Normalization
    for layer in model.layers:
        if isinstance(layer, layers.Normalization):
            layer.adapt(X_train)
            break
            
    # Callback
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    history = model.fit(
        X_train, y_train_enc,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_enc),
        sample_weight=sample_weight,
        callbacks=[es_callback],
        verbose=2
    )
    
    # Return model, history, and the encoder (needed for evaluation/inference)
    # We should attach encoder to model or return tuple.
    # The signature only mentions returning model/history.
    # But without encoder we can't decode predictions.
    # We'll attach it to the model object for convenience or return tuple.
    model.label_encoder = label_encoder
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    returns accuracy and all other metrics you consider useful.
    """
    # Predict
    y_pred_prob = model.predict(X_test)
    y_pred_idx = np.argmax(y_pred_prob, axis=1)
    
    if hasattr(model, 'label_encoder'):
        # OneHotEncoder.categories_ is a list of arrays (one per feature)
        # We index into categories_[0] using the predicted indices
        y_pred = model.label_encoder.categories_[0][y_pred_idx]
    else:
        # Fallback if encoder somehow lost, assume user handles it or we fail gracefully
        print("Warning: Label encoder not found on model.")
        return {}

    from sklearn.metrics import classification_report, accuracy_score, f1_score
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")
    print(report)
    
    return {
        "accuracy": acc,
        "f1_score": f1,
        "report": report
    }

def save_model(model, save_dir):
    """
    saving/loading utilities (DL via Keras).
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "dense_model.h5"))
    if hasattr(model, 'label_encoder'):
        joblib.dump(model.label_encoder, os.path.join(save_dir, "label_encoder.joblib"))

def load_model(save_dir):
    """
    loading utilities.
    """
    path = os.path.join(save_dir, "dense_model.h5")
    encoder_path = os.path.join(save_dir, "label_encoder.joblib")
    
    if os.path.exists(path):
        model = models.load_model(path)
        if os.path.exists(encoder_path):
            model.label_encoder = joblib.load(encoder_path)
        return model
    return None
