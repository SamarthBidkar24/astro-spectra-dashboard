
import os
import glob
import re
import pathlib
import hashlib
import tarfile
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

def comp_sha256(file_name):
    """
    Compute the SHA256 hash of a file.
    """
    hash_sha256 = hashlib.sha256()
    with pathlib.Path(file_name).open(mode="rb") as f_temp:
        for _seq in iter(lambda: f_temp.read(65536), b""):
            hash_sha256.update(_seq)
    return hash_sha256.hexdigest()

def load_raw_data():
    """
    loads the raw spectra exactly as done in the original notebooks.
    Downloads to data/lvl0 if not present.
    """
    # Assuming execution from project root or relative paths needed.
    # The request says "In Project-Asteroid-Spectra, create a src/ folder..."
    # We will use relative path "data/lvl0" from CWD or absolute.
    # Let's use absolute based on file location.
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data", "lvl0")
    
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    files_to_dl = {
        'file1': {'url': 'http://smass.mit.edu/data/smass/Bus.Taxonomy.txt',
                  'sha256': '0ce970a6972dd7c49d512848b9736d00b621c9d6395a035bd1b4f3780d4b56c6'},
        'file2': {'url': 'http://smass.mit.edu/data/smass/smass2data.tar.gz',
                  'sha256': 'dacf575eb1403c08bdfbffcd5dbfe12503a588e09b04ed19cc4572584a57fa97'}
    }
    
    for dl_key in files_to_dl:
        url = files_to_dl[dl_key]["url"]
        filename = os.path.join(data_dir, url.split("/")[-1])
        
        if not os.path.isfile(filename):
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, filename)
            
            file_hash = comp_sha256(filename)
            if file_hash != files_to_dl[dl_key]["sha256"]:
                print(f"Hash mismatch for {filename}!")
            else:
                print(f"Verified {filename}")
                
    # Extract tar
    tar_path = os.path.join(data_dir, "smass2data.tar.gz")
    if os.path.exists(tar_path):
        if not os.path.exists(os.path.join(data_dir, "smass2")):
            print("Extracting tar file...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(data_dir)

def build_clean_dataset():
    """
    does all parsing and enrichment and saves a single cleaned dataset file, e.g. data/asteroid_clean.csv.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    lvl0_dir = os.path.join(project_root, "data", "lvl0")
    out_dir = os.path.join(project_root, "data")
    
    # Ensure raw data is loaded
    load_raw_data()
    
    print("Parsing data...")
    spectra_filepaths = sorted(glob.glob(os.path.join(lvl0_dir, "smass2", "*spfit*")))
    
    if not spectra_filepaths:
        print("No spectra files found. Check download.")
        return

    # Create DataFrames for file paths
    # Logic from notebook 2
    des_file_paths = spectra_filepaths[:-8]
    non_file_paths = spectra_filepaths[-8:]
    
    des_file_paths_df = pd.DataFrame(des_file_paths, columns=["FilePath"])
    non_file_paths_df = pd.DataFrame(non_file_paths, columns=["FilePath"])
    
    # Regex from previous fix
    des_file_paths_df["DesNr"] = des_file_paths_df["FilePath"].apply(
        lambda x: int(re.search(r'a(.*).spfit', pathlib.Path(x).name).group(1))
    )
    non_file_paths_df["DesNr"] = non_file_paths_df["FilePath"].apply(
        lambda x: re.search(r'au(.*).spfit', pathlib.Path(x).name).group(1)
    )
    
    # Read classification
    bus_tax_path = os.path.join(lvl0_dir, "Bus.Taxonomy.txt")
    asteroid_class_df = pd.read_csv(bus_tax_path, skiprows=21, sep="\t",
                                    names=["Name", "Tholen_Class", "Bus_Class", "unknown1", "unknown2"])
    asteroid_class_df["Name"] = asteroid_class_df["Name"].str.strip()
    
    # Split designated/non-designated
    des_ast_class_df = asteroid_class_df[:1403].copy()
    non_ast_class_df = asteroid_class_df[1403:].copy()
    
    des_ast_class_df["DesNr"] = des_ast_class_df["Name"].apply(lambda x: int(x.split(" ")[0]))
    non_ast_class_df["DesNr"] = non_ast_class_df["Name"].apply(lambda x: x.replace(" ", ""))
    
    # Merge
    des_join = des_ast_class_df.merge(des_file_paths_df, on="DesNr")
    non_join = non_ast_class_df.merge(non_file_paths_df, on="DesNr")
    asteroids_df = pd.concat([des_join, non_join], axis=0).reset_index(drop=True)
    
    # Cleanup
    asteroids_df.drop(columns=["Tholen_Class", "unknown1", "unknown2"], inplace=True)
    asteroids_df.dropna(subset=["Bus_Class"], inplace=True)
    
    # Read spectra
    def read_spec(path):
        return pd.read_csv(path, sep="\t", names=["Wavelength_in_microm", "Reflectance_norm550nm"])
        
    asteroids_df["SpectrumDF"] = asteroids_df["FilePath"].apply(read_spec)
    asteroids_df["DesNr"] = asteroids_df["DesNr"].astype(str)
    
    # Enrichment
    print("Enriching data...")
    bus_to_main = {
        'A': 'Other', 'B': 'C', 'C': 'C', 'Cb': 'C', 'Cg': 'C', 'Cgh': 'C', 'Ch': 'C',
        'D': 'Other', 'K': 'Other', 'L': 'Other', 'Ld': 'Other', 'O': 'Other', 'R': 'Other',
        'S': 'S', 'Sa': 'S', 'Sk': 'S', 'Sl': 'S', 'Sq': 'S', 'Sr': 'S',
        'T': 'Other', 'V': 'Other', 'X': 'X', 'Xc': 'X', 'Xe': 'X', 'Xk': 'X'
    }
    asteroids_df["Main_Group"] = asteroids_df["Bus_Class"].apply(lambda x: bus_to_main.get(x, "None"))
    
    # Filter 'None' if any
    asteroids_df = asteroids_df[asteroids_df["Main_Group"] != "None"]
    
    # Save
    out_path = os.path.join(out_dir, "asteroid_clean.pkl")
    asteroids_df.to_pickle(out_path)
    print(f"Saved clean dataset to {out_path}")
    return asteroids_df

def get_train_test_split(test_size=0.2, random_state=42):
    """
    loads the cleaned dataset, prepares features and labels, and returns X_train, X_test, y_train, y_test 
    with the same preprocessing as in the notebooks.
    
    Returns raw features (not scaled) because notebooks apply scaling differently based on model type 
    (StandardScaler for SVM, Normalization layer for Dense/Conv).
    If scaling is universally applied before splitting in notebooks, I'd calculate it here, but 
    notebook 8/9 adapt normalization on training data after split.
    Notebook 5/7 fit scaler on training data after split.
    
    So this function returns RAW X/y split.
    Feature extraction: Flatten SpectrumDF -> 'Reflectance_norm550nm'
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    clean_path = os.path.join(project_root, "data", "asteroid_clean.pkl")
    
    if not os.path.exists(clean_path):
        print("Clean dataset not found, building...")
        build_clean_dataset()
        
    df = pd.read_pickle(clean_path)
    
    # Features
    X = np.array([k["Reflectance_norm550nm"].tolist() for k in df["SpectrumDF"]])
    # Labels
    y = np.array(df["Main_Group"].to_list())
    
    # Stratified Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test
    
    return None, None, None, None
