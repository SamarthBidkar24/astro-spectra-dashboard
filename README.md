
# Asteroid Spectra Classification 🌠

## Introduction
Asteroid Spectra Classification is a machine learning project designed to classify asteroids based on their reflectance spectra. Asteroids are remnants from the early solar system, and understanding their composition helps scientists learn about the formation of planets.

This project automates the classification of asteroids into major taxonomic types (C, S, X, and Other) using three different machine learning models:
1.  **Support Vector Machine (SVM)**
2.  **Dense Neural Network (DNN)**
3.  **1D Convolutional Neural Network (ConvNet)**

We also provide an interactive **Streamlit Dashboard** to visualize spectra and see real-time predictions from all models.

---

## Dataset & Features 📊
The dataset comes from the **SMASS II (Small Main-Belt Asteroid Spectroscopic Survey)**.
*   **Input**: Reflectance spectrum (light intensity reflected at different wavelengths).
*   **Wavelength Range**: Approximately 0.44 µm to 0.92 µm (Visible to Near-Infrared).
*   **Output**: Asteroid Class (C-type, S-type, X-type, or Other).

The raw data is processed to:
1.  Normalize reflectance values.
2.  Interpolate spectra to a common wavelength grid.
3.  Group complex sub-classes into the four main categories.

---

## Models Explained 🧠

### 1. Support Vector Machine (SVM)
*   **Role**: A robust baseline model.
*   **How it works**: It finds the best "hyperplane" (boundary) that separates different classes of asteroids in a high-dimensional space.
*   **Why use it?**: SVMs work very well on smaller datasets and provide a strong benchmark for comparison.

### 2. Dense Neural Network (DNN)
*   **Role**: A simple deep learning model.
*   **How it works**: It treats the entire spectrum as a single vector of numbers and learns non-linear patterns through multiple layers of neurons.
*   **Why use it?**: It can capture more complex relationships than an SVM but is simpler than a ConvNet.

### 3. 1D Convolutional Neural Network (ConvNet)
*   **Role**: The advanced model.
*   **How it works**: It slides "filters" across the spectrum to detect local shapes, like absorption bands (dips in the graph), regardless of where they appear.
*   **Why use it?**: This architecture is specifically designed for sequential data like spectra and often achieves the highest accuracy.

---

## Evaluation Metrics 📈
We evaluate models using:
*   **Accuracy**: The percentage of correctly classified asteroids.
*   **F1-Score**: A balanced metric that considers both Precision (false positives) and Recall (false negatives).
*   **Confusion Matrix**: A chart showing where the model gets confused (e.g., mistaking an 'X' type for a 'C' type).

All evaluation results are saved in the `metrics/` and `plots/` folders.

---

## Streamlit Dashboard 🖥️
The project includes a user-friendly web app with four tabs:

1.  **Predict**:
    *   **Input**: Select a sample asteroid or upload your own CSV file.
    *   **Visualize**: See the spectrum plot.
    *   **Results**: View a table comparing predictions from all three models, including confidence scores.
    *   **Charts**: Probability bar charts for each model.

2.  **Metrics**:
    *   Displays the overall accuracy and F1-scores for all models.
    *   Provides detailed per-class performance tables.

3.  **Diagnostics**:
    *   Shows the confusion matrices to visualize model errors.
    *   (Optional) Displays training history curves if available.

4.  **About**:
    *   Explains the project background and model details.

---

## How to Run the Project 🚀

### 1. Setup Environment
Ensure you have Python installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```
*(Dependencies include: numpy, pandas, scikit-learn, tensorflow, matplotlib, seaborn, streamlit, altair)*

### 2. Train Models
To train all three models from scratch and save them to the `models/` directory:

```bash
python train_all_models.py
```

### 3. Evaluate Models
To generate performance metrics and confusion matrix plots without retraining:

```bash
python evaluate_and_plot.py
```
This will create JSON files in `metrics/` and images in `plots/`.

### 4. Run the Dashboard
To launch the interactive application:

```bash
streamlit run app.py
```
This will open the dashboard in your default web browser (usually at `http://localhost:8501`).

---

**Developed for Asteroid Classification Major Project.**
