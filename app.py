
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import matplotlib.pyplot as plt
from scipy import interpolate
import tensorflow as tf
from tensorflow import keras

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
# Attempt to import SVMWrapper for unpickling
try:
    from src.models_svm import SVMWrapper
except ImportError:
    # If unpickling works without it (sometimes if structure is simple), great.
    # Otherwise, we might define a dummy class if needed, but since we append src, it should work.
    pass

# Page Config
st.set_page_config(page_title="Asteroid Spectra Classifier", layout="wide")

# ==============================================================================
# 1. Utilities and Loading
# ==============================================================================

@st.cache_resource
def load_resources():
    """
    Load models, scaler, class map, and clean dataset.
    """
    resources = {}
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # 1. Scaler
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        resources['scaler'] = joblib.load(scaler_path)
    else:
        st.error("Scaler not found in models/scaler.pkl")
        return None

    # 2. Class Map
    map_path = os.path.join(models_dir, "class_map.json")
    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            resources['class_map'] = json.load(f)
    else:
        st.error("Class map not found in models/class_map.json")
        return None

    # 3. Models
    # SVM
    svm_path = os.path.join(models_dir, "svm_multiclass.pkl")
    if os.path.exists(svm_path):
        resources['svm'] = joblib.load(svm_path)
    
    # Dense
    dense_path = os.path.join(models_dir, "dense_multiclass.h5")
    if os.path.exists(dense_path):
        resources['dense'] = keras.models.load_model(dense_path)
        
    # ConvNet
    conv_path = os.path.join(models_dir, "convnet_multiclass.h5")
    if os.path.exists(conv_path):
        resources['convnet'] = keras.models.load_model(conv_path)
        
    # 4. Clean Data (for samples and grid)
    clean_path = os.path.join(data_dir, "asteroid_clean.pkl")
    if os.path.exists(clean_path):
        resources['data'] = pd.read_pickle(clean_path)
        # Extract canonical grid from first sample
        if not resources['data'].empty:
            # Assuming 'SpectrumDF' column contains DataFrames
            first_spec = resources['data'].iloc[0]["SpectrumDF"]
            resources['grid'] = first_spec["Wavelength_in_microm"].values
            resources['grid_len'] = len(resources['grid'])
    
    return resources

def interpolate_spectrum(input_wavelengths, input_reflectances, target_grid):
    """
    Interpolate input spectrum to target wavelength grid.
    """
    # Create interpolator
    # Use bounds_error=False and fill_value="extrapolate" or fixed?
    # Extrapolation might be dangerous if input range is small.
    # But usually spectra cover similar ranges (0.4 to 0.9 microns).
    f = interpolate.interp1d(input_wavelengths, input_reflectances, kind='linear', 
                             bounds_error=False, fill_value="extrapolate")
    return f(target_grid)

# Load everything
res = load_resources()
if not res:
    st.image("https://media.giphy.com/media/26hkhKd9CQzzRE7de/giphy.gif")
    st.stop()

# Short names
scaler = res['scaler']
class_map = res['class_map']
model_svm = res.get('svm')
model_dense = res.get('dense')
model_conv = res.get('convnet')
clean_df = res.get('data')
target_grid = res.get('grid')

# ==============================================================================
# 2. Sidebar
# ==============================================================================
st.sidebar.title("Input Spectrum")

input_mode = st.sidebar.radio("Input Source", ["Sample from Dataset", "Upload CSV"])

selected_spectrum_name = None
x_input = None # Wavelengths
y_input = None # Reflectance

if input_mode == "Sample from Dataset":
    if clean_df is not None:
        # Create a label for selection: "Designation (Class)"
        clean_df['label'] = clean_df.apply(lambda r: f"{r['DesNr']} ({r['Main_Group']})", axis=1)
        
        sid = st.sidebar.selectbox("Select Asteroid", clean_df['label'].tolist())
        
        # Get data
        row = clean_df[clean_df['label'] == sid].iloc[0]
        spec_df = row['SpectrumDF']
        x_input = spec_df["Wavelength_in_microm"].values
        y_input = spec_df["Reflectance_norm550nm"].values
        selected_spectrum_name = sid
    else:
        st.sidebar.error("Dataset not available.")

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (cols: wavelength, reflectance)", type="csv")
    if uploaded_file:
        try:
            udf = pd.read_csv(uploaded_file)
            # Normalize column names lower case
            udf.columns = [c.lower() for c in udf.columns]
            
            # Look for suitable columns
            w_col = next((c for c in udf.columns if 'wave' in c), None)
            r_col = next((c for c in udf.columns if 'refl' in c or 'flux' in c), None)
            
            if w_col and r_col:
                x_input = udf[w_col].values
                y_input = udf[r_col].values
                selected_spectrum_name = uploaded_file.name
            else:
                st.sidebar.error("Could not identify 'wavelength' and 'reflectance' columns.")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

# ==============================================================================
# 3. Main Layout
# ==============================================================================
st.title("Asteroid Spectra Classification")

tabs = st.tabs(["Predict", "Metrics", "Diagnostics", "About"])

# --- Tab: Predict ---
with tabs[0]:
    st.info(
        "**Step 1:** Choose a sample spectrum or upload a wavelength/reflectance CSV. "
        "**Step 2:** Scroll down to see predictions from all models. "
        "**Step 3:** Use the **Metrics** and **Diagnostics** tabs to understand model performance."
    )
    
    if x_input is not None and y_input is not None:
        st.subheader(f"Analyzing: {selected_spectrum_name}")
        
        # 1. Plot
        # Use Altair for better labels "Input Reflectance Spectrum", "Wavelength (µm)", "Relative Reflectance"
        import altair as alt
        chart_data = pd.DataFrame({"Wavelength": x_input, "Reflectance": y_input})
        
        c_spec = alt.Chart(chart_data).mark_line().encode(
            x=alt.X('Wavelength', title='Wavelength (µm)'),
            y=alt.Y('Reflectance', title='Relative Reflectance'),
            tooltip=['Wavelength', 'Reflectance']
        ).properties(
            title='Input Reflectance Spectrum',
            width='container'
        )
        st.altair_chart(c_spec, use_container_width=True)
        
        # 2. Preprocess (Interpolate)
        if target_grid is not None:
            y_interp = interpolate_spectrum(x_input, y_input, target_grid)
            
            # Prepare for models
            # SVM: expect (1, n_features) raw (wrapper handles scaling)
            X_in = y_interp.reshape(1, -1)
            
            # 3. Predict
            results = []
            
            # Helper to extract top 2
            def get_top_k(probs, classes, k=2):
                if probs is None:
                    return None
                # Sort indices descending
                top_idx = np.argsort(probs)[-k:][::-1]
                res = []
                for idx in top_idx:
                    res.append((classes[idx], probs[idx]))
                return res

            # --- SVM ---
            if model_svm:
                try:
                    pred_cls = model_svm.predict(X_in)[0]
                    # Attempt probability
                    try:
                        probs = model_svm.model.predict_proba(model_svm.scaler.transform(X_in))[0]
                        top2 = get_top_k(probs, model_svm.model.classes_, k=2)
                        
                        conf = f"{top2[0][1]:.4f}"
                        sec_cls = top2[1][0] if len(top2) > 1 else "N/A"
                        sec_conf = f"{top2[1][1]:.4f}" if len(top2) > 1 else "N/A"
                        
                        all_classes = model_svm.model.classes_
                    except:
                        probs = None
                        conf = "N/A"
                        sec_cls = "N/A"
                        sec_conf = "N/A"
                        all_classes = []
                        
                    results.append({
                        "Model": "SVM",
                        "Predicted Class": pred_cls,
                        "Confidence": conf,
                        "Second Best Class": sec_cls,
                        "Second Best Probability": sec_conf,
                        "Probs": probs,
                        "Classes": all_classes
                    })
                except Exception as e:
                    st.error(f"SVM Error: {e}")

            # --- Dense ---
            if model_dense:
                try:
                    probs = model_dense.predict(X_in, verbose=0)[0]
                    top2 = get_top_k(probs, class_map, k=2)
                    
                    pred_cls = top2[0][0]
                    conf = f"{top2[0][1]:.4f}"
                    sec_cls = top2[1][0] if len(top2) > 1 else "N/A"
                    sec_conf = f"{top2[1][1]:.4f}" if len(top2) > 1 else "N/A"
                    
                    results.append({
                        "Model": "Dense NN",
                        "Predicted Class": pred_cls,
                        "Confidence": conf,
                        "Second Best Class": sec_cls,
                        "Second Best Probability": sec_conf,
                        "Probs": probs,
                        "Classes": class_map
                    })
                except Exception as e:
                    st.error(f"Dense NN Error: {e}")

            # --- ConvNet ---
            if model_conv:
                try:
                    X_conv = np.expand_dims(X_in, axis=2)
                    probs = model_conv.predict(X_conv, verbose=0)[0]
                    top2 = get_top_k(probs, class_map, k=2)
                    
                    pred_cls = top2[0][0]
                    conf = f"{top2[0][1]:.4f}"
                    sec_cls = top2[1][0] if len(top2) > 1 else "N/A"
                    sec_conf = f"{top2[1][1]:.4f}" if len(top2) > 1 else "N/A"
                    
                    results.append({
                        "Model": "ConvNet",
                        "Predicted Class": pred_cls,
                        "Confidence": conf,
                        "Second Best Class": sec_cls,
                        "Second Best Probability": sec_conf,
                        "Probs": probs,
                        "Classes": class_map
                    })
                except Exception as e:
                    st.error(f"ConvNet Error: {e}")

            # Display Table
            if results:
                res_df = pd.DataFrame(results)[["Model", "Predicted Class", "Confidence", "Second Best Class", "Second Best Probability"]]
                st.table(res_df)
                
                # Display Probability Bars
                st.subheader("Probability Distributions")
                
                # Fixed Color Map
                color_map = {
                    "C": "#333333",   # Dark Grey
                    "S": "#D35400",   # Red/Orange
                    "X": "#7F8C8D",   # Grey
                    "Other": "#8E44AD" # Purple
                }
                
                cols = st.columns(len(results))
                for i, res_item in enumerate(results):
                    with cols[i]:
                        # Title for chart
                        model_name = res_item["Model"]
                        
                        if res_item["Probs"] is not None:
                            # Create DF
                            prob_df = pd.DataFrame({
                                "Class": res_item["Classes"],
                                "Probability": res_item["Probs"]
                            })
                            # Add Color Column
                            prob_df["Color"] = prob_df["Class"].apply(lambda c: color_map.get(c, "#000000"))
                            
                            c = alt.Chart(prob_df).mark_bar().encode(
                                x=alt.X('Class', title='Asteroid Class', sort=None),
                                y=alt.Y('Probability', title='Probability', scale=alt.Scale(domain=[0, 1])),
                                color=alt.Color('Class', scale=alt.Scale(
                                    domain=list(color_map.keys()),
                                    range=list(color_map.values())
                                ), legend=None),
                                tooltip=['Class', 'Probability']
                            ).properties(
                                title=f"Class Probabilities – {model_name}",
                                height=200
                            )
                            
                            st.altair_chart(c, use_container_width=True)
                            
                        else:
                            st.write("No probability data.")
        else:
            st.warning("Target grid not available. Cannot interpolate.")
            
    else:
        st.info("Please select or upload a spectrum to predict.")

# --- Tab: Metrics ---
with tabs[1]:
    st.header("Model Performance Metrics")
    
    metrics_data = []
    
    # helper to load json
    def load_metrics(name, fname):
        path = os.path.join("metrics", fname)
        if os.path.exists(path):
            with open(path, 'r') as f:
                d = json.load(f)
                d['model'] = name
                return d
        return None
        
    m_svm = load_metrics("SVM", "svm_metrics.json")
    m_dense = load_metrics("Dense NN", "dense_metrics.json")
    m_conv = load_metrics("ConvNet", "convnet_metrics.json")
    
    all_metrics = [m for m in [m_svm, m_dense, m_conv] if m]
    
    if all_metrics:
        summary = []
        for m in all_metrics:
            summary.append({
                "Model": m['model'],
                "Accuracy": f"{m['overall_accuracy']:.4f}",
                "Macro F1": f"{m['macro_metrics']['f1']:.4f}"
            })
        st.table(pd.DataFrame(summary))
        
        # Per Class Expander
        with st.expander("Detailed Per-Class Metrics"):
            for m in all_metrics:
                st.subheader(m['model'])
                per_class = pd.DataFrame(m['per_class_metrics']).T
                st.dataframe(per_class.style.format("{:.4f}"))
    else:
        st.warning("No metrics files found in metrics/")

# --- Tab: Diagnostics ---
with tabs[2]:
    st.header("Training Diagnostics & Confusion Matrices")
    
    plot_dir = "plots"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SVM")
        if os.path.exists(f"{plot_dir}/svm_confusion.png"):
            st.image(f"{plot_dir}/svm_confusion.png", caption="SVM Confusion Matrix")
            
    with col2:
        st.subheader("Dense NN")
        if os.path.exists(f"{plot_dir}/dense_confusion.png"):
            st.image(f"{plot_dir}/dense_confusion.png", caption="Dense NN Confusion Matrix")
        if os.path.exists(f"{plot_dir}/dense_history.png"):
            st.image(f"{plot_dir}/dense_history.png", caption="Dense NN Training History")
            
    st.subheader("ConvNet")
    if os.path.exists(f"{plot_dir}/convnet_confusion.png"):
        st.image(f"{plot_dir}/convnet_confusion.png", caption="ConvNet Confusion Matrix")
    if os.path.exists(f"{plot_dir}/convnet_history.png"):
        st.image(f"{plot_dir}/convnet_history.png", caption="ConvNet Training History")

# --- Tab: About ---
with tabs[3]:
    st.header("About Asteroid Spectral Classification")
    
    st.markdown("""
    **Asteroid Classes:**
    Asteroids are classified based on their reflection spectra (how they reflect sunlight across different wavelengths). 
    Common classes include **C-type** (carbonaceous, dark), **S-type** (silicaceous, stony), **X-type** (metallic/other), and various others. 
    Identifying these classes helps understanding the composition and history of our solar system.

    **Why Multiple Models?**
    We use three different approaches:
    1. **SVM (Support Vector Machine):** A robust classical machine learning algorithm effective for smaller datasets and high-dimensional spaces.
    2. **Dense Neural Network:** A simple deep learning model that learns non-linear patterns across the entire spectral range.
    3. **1D ConvNet (Convolutional Neural Network):** Specialized for sequential data like spectra, it can detect local features (absorption bands) regardless of their absolute position, often providing superior performance.

    **Interpreting Results:**
    - **Confusion Matrix:** Shows where the model makes mistakes. Ideally, the diagonal is bright (correct predictions). Off-diagonal elements show misclassifications.
    - **Probabilities:** Represent the model's confidence. High confidence doesn't always guarantee correctness, but low confidence suggests ambiguity (e.g., an asteroid with properties of strictly different classes).
    """)

