
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_data
from src.preprocessing import get_feature_matrix, get_labels

if __name__ == "__main__":
    core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"Project root: {core_path}")

    # Load data
    try:
        asteroids_df = load_data(core_path)
        print(f"Data loaded successfully. Shape: {asteroids_df.shape}")
        
        # Check features
        X = get_feature_matrix(asteroids_df)
        print(f"Feature matrix shape: {X.shape}")
        
        # Check labels
        y = get_labels(asteroids_df)
        print(f"Labels shape: {y.shape}")
        print(f"Sample labels: {y[:5]}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
