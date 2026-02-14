
from sklearn import svm
import tensorflow as keras
from tensorflow.keras import layers, models

def get_svm_model(class_weight=None, C=100, kernel='rbf'):
    """
    Returns an SVM classifier.
    """
    clf = svm.SVC(kernel=kernel, class_weight=class_weight, C=C)
    return clf

def create_dense_model(input_shape, n_outputs=4):
    """
    Creates a simple Dense-layer-based neural network.
    Structure based on 8_dl_dense_multiclass.ipynb
    """
    input_layer = layers.Input(shape=input_shape)
    
    # Normalization layer
    # Note: caller must call model.layers[1].adapt(X_train) before training!
    # Or we handle it differently. Here we just define the structure.
    normalizer = layers.Normalization(axis=1)
    
    hidden_layer = normalizer(input_layer)
    hidden_layer = layers.Dense(25)(hidden_layer)
    hidden_layer = layers.ReLU()(hidden_layer)
    
    hidden_layer = layers.Dense(10)(hidden_layer)
    hidden_layer = layers.ReLU()(hidden_layer)
    
    output_layer = layers.Dense(n_outputs, activation="softmax")(hidden_layer)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_conv_model(input_shape, n_outputs=4):
    """
    Creates a Convolutional Neural Network.
    Structure based on 9_dl_convnet_multiclass.ipynb
    """
    input_layer = layers.Input(shape=input_shape)
    
    normalizer = layers.Normalization(axis=1)
    norm_layer = normalizer(input_layer)
    
    hidden_layer = layers.Conv1D(filters=32, activation="relu", kernel_size=3)(norm_layer)
    hidden_layer = layers.MaxPooling1D(pool_size=2)(hidden_layer)
    
    hidden_layer = layers.Conv1D(filters=64, activation="relu", kernel_size=5)(hidden_layer)
    hidden_layer = layers.MaxPooling1D(pool_size=2)(hidden_layer)
    
    hidden_layer = layers.Flatten()(hidden_layer)
    hidden_layer = layers.Dense(16, activation="relu")(hidden_layer)
    
    output_layer = layers.Dense(n_outputs, activation="softmax")(hidden_layer)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
