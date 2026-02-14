
import os
import sys
import json
import joblib
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_processing import load_data
from src.preprocessing import get_feature_matrix, get_labels, stratified_split

# Page Config
st.set_page_config(page_title="Asteroid Spectra Classification", layout="wide")

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
METRICS_DIR = os.path.join(ROOT_DIR, "metrics")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")

# Helper Functions
@st.cache_resource
def load_cached_data():
    return load_data(ROOT_DIR)

@st.cache_resource
def load_svm_model():
    path = os.path.join(MODELS_DIR, "svm_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    if os.path.exists(path) and os.path.exists(scaler_path):
        return joblib.load(path), joblib.load(scaler_path)
    return None, None

@st.cache_resource
def load_dense_model():
    path = os.path.join(MODELS_DIR, "dense_model")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder_dense.joblib")
    if os.path.exists(path) and os.path.exists(encoder_path):
        return tf.keras.models.load_model(path), joblib.load(encoder_path)
    return None, None

@st.cache_resource
def load_conv_model():
    path = os.path.join(MODELS_DIR, "conv_model")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder_conv.joblib")
    if os.path.exists(path) and os.path.exists(encoder_path):
        return tf.keras.models.load_model(path), joblib.load(encoder_path)
    return None, None

# Main Layout
st.title("Asteroid Spectral Classification Dashboard")
st.markdown("""
This dashboard allows you to explore different Machine Learning models trained to classify asteroids based on their reflectance spectra.
""")

# Sidebar
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox("Choose a Model", ["SVM", "Dense Neural Network", "Convolutional Neural Network"])

# Helper to get paths based on selection
if model_type == "SVM":
    metrics_file = "svm_metrics.json"
    cm_plot_file = "svm_confusion_matrix.png"
    loss_plot_file = None
elif model_type == "Dense Neural Network":
    metrics_file = "dense_metrics.json"
    cm_plot_file = "dense_confusion_matrix.png"
    loss_plot_file = "dense_loss.png"
else:
    metrics_file = "conv_metrics.json"
    cm_plot_file = "conv_confusion_matrix.png"
    loss_plot_file = "conv_loss.png"

# Display Metrics
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model Performance")
    metrics_path = os.path.join(METRICS_DIR, metrics_file)
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        st.metric("F1 Score (Weighted)", f"{metrics.get('f1_score', 0):.4f}")
        
        if "best_params" in metrics:
            st.write("Best Parameters:", metrics["best_params"])
        if "epochs" in metrics:
            st.write(f"Trained for {metrics['epochs']} epochs")
            
        st.text("Classification Report:")
        st.text(metrics.get("classification_report", ""))
    else:
        st.warning("Metrics file not found. Have you trained the model?")

with col2:
    st.subheader("Visualizations")
    tab1, tab2 = st.tabs(["Confusion Matrix", "Loss Curve"])
    
    with tab1:
        cm_path = os.path.join(PLOTS_DIR, cm_plot_file)
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix")
        else:
            st.info("Confusion matrix plot not available.")
            
    with tab2:
        if loss_plot_file:
            loss_path = os.path.join(PLOTS_DIR, loss_plot_file)
            if os.path.exists(loss_path):
                st.image(loss_path, caption="Training Loss")
            else:
                st.info("Loss plot not available.")
        else:
            st.write("Not applicable for this model.")

# Inference Section
st.divider()
st.header("Real-time Inference")

# Load Data for Sampling
try:
    df = load_cached_data()
    X = get_feature_matrix(df)
    y = get_labels(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y) # Use test split for demo
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if st.button("Sample Random Spectrum"):
    idx = random.randint(0, len(X_test) - 1)
    st.session_state['sample_idx'] = idx

if 'sample_idx' in st.session_state:
    idx = st.session_state['sample_idx']
    sample_spectrum = X_test[idx] # Shape (49,) or (49, 1) depending on how preprocessing handles it? 
    # preprocessing.get_feature_matrix returns (N, 49).
    true_label = y_test[idx]
    
    # Plot Spectrum
    fig, ax = plt.subplots(figsize=(10, 4))
    wavelengths = np.linspace(0.44, 0.92, 49) # Approx mapping based on NB 2
    ax.plot(wavelengths, sample_spectrum, label="Spectrum")
    ax.set_xlabel("Wavelength (microns)")
    ax.set_ylabel("Reflectance")
    ax.set_title(f"Sample Spectrum (True Class: {true_label})")
    ax.legend()
    st.pyplot(fig)
    
    # Prediction
    st.subheader("Prediction Result")
    
    if model_type == "SVM":
        model, scaler = load_svm_model()
        if model and scaler:
            # Preprocess
            sample_scaled = scaler.transform(sample_spectrum.reshape(1, -1))
            pred = model.predict(sample_scaled)[0]
            st.success(f"Predicted Class: {pred}")
        else:
            st.error("Model or Scaler not found.")
            
    elif model_type == "Dense Neural Network":
        model, encoder = load_dense_model()
        if model and encoder:
            # Preprocess: Dense model expects (1, 49) and handles normalization internally IF adapted properly.
            # But wait, in train_dense.py we adapted the normalization layer.
            # So we pass raw input.
            sample_input = sample_spectrum.reshape(1, -1)
            pred_prob = model.predict(sample_input, verbose=0)
            pred_idx = np.argmax(pred_prob)
            pred_label = encoder.inverse_transform([pred_idx])[0]
            
            st.success(f"Predicted Class: {pred_label}")
            
            # Show probabilities
            st.write("Probabilities:")
            prob_df = pd.DataFrame(pred_prob, columns=encoder.classes_)
            st.bar_chart(prob_df.T)
        else:
            st.error("Model or Encoder not found.")
            
    elif model_type == "Convolutional Neural Network":
        model, encoder = load_conv_model()
        if model and encoder:
            # Preprocess: Conv expects (1, 49, 1)
            sample_input = sample_spectrum.reshape(1, 49, 1)
            pred_prob = model.predict(sample_input, verbose=0)
            pred_idx = np.argmax(pred_prob)
            pred_label = encoder.inverse_transform([pred_idx])[0]
            
            st.success(f"Predicted Class: {pred_label}")
            
            st.write("Probabilities:")
            prob_df = pd.DataFrame(pred_prob, columns=encoder.classes_)
            st.bar_chart(prob_df.T)
        else:
            st.error("Model or Encoder not found.")
