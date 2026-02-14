
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

def get_feature_matrix(asteroids_df):
    """
    Flattens the SpectrumDF column into a numpy array.
    """
    # Allocate the spectra to one array
    asteroids_X = np.array([k["Reflectance_norm550nm"].tolist() for k in asteroids_df["SpectrumDF"]])
    return asteroids_X

def get_labels(asteroids_df):
    """
    Returns the Main_Group labels.
    """
    return np.array(asteroids_df["Main_Group"].to_list())

def encode_labels_binary(asteroids_df, target_class="X"):
    """
    Creates binary labels (1 for target_class, 0 for others).
    """
    return asteroids_df["Main_Group"].apply(lambda x: 1 if x == target_class else 0).to_numpy()

def encode_labels_multiclass(y):
    """
    One-hot encodes the labels.
    Returns the encoder and the encoded labels.
    """
    label_encoder = preprocessing.OneHotEncoder(sparse=True)
    asteroids_oh_y = label_encoder.fit_transform(y.reshape(-1, 1)).toarray()
    return label_encoder, asteroids_oh_y

def scale_data(X_train, X_test):
    """
    Scales data using StandardScaler fitted on training data.
    """
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

def stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Performs stratified shuffle split.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test
    return None, None, None, None
