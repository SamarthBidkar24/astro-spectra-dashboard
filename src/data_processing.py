
import glob
import os
import pathlib
import tarfile
import urllib.request
import urllib.parse
import re

import pandas as pd
import numpy as np

# Adjust import based on execution context
try:
    from src.utils import comp_sha256
except ImportError:
    from utils import comp_sha256

def download_data(core_path):
    """
    Downloads required asteroid taxonomy data.
    Based on logic from 1_data_fetch.ipynb
    """
    print("Step 1: Data Fetching")
    
    # Create the level0 data directory
    pathlib.Path(os.path.join(core_path, "data/lvl0/")).mkdir(parents=True, exist_ok=True)
    
    # Set a dictionary that contains the taxonomy classification data and corresponding sha256 values
    files_to_dl = {
        'file1': {'url': 'http://smass.mit.edu/data/smass/Bus.Taxonomy.txt',
                  'sha256': '0ce970a6972dd7c49d512848b9736d00b621c9d6395a035bd1b4f3780d4b56c6'},
        'file2': {'url': 'http://smass.mit.edu/data/smass/smass2data.tar.gz',
                  'sha256': 'dacf575eb1403c08bdfbffcd5dbfe12503a588e09b04ed19cc4572584a57fa97'}
    }
    
    for dl_key in files_to_dl:
        # Get the URL and create a download filepath by splitting it at the last "/"
        split = urllib.parse.urlsplit(files_to_dl[dl_key]["url"])
        filename = pathlib.Path(os.path.join(core_path, "data/lvl0/", split.path.split("/")[-1]))
        
        # Download file if it is not available
        if not filename.is_file():
            print(f"Downloading now: {files_to_dl[dl_key]['url']}")
            
            # Download file and retrieve the created filepath
            downl_file_path, _ = urllib.request.urlretrieve(url=files_to_dl[dl_key]["url"],
                                                            filename=filename)
            
            # Compute and compare the hash value
            tax_hash = comp_sha256(downl_file_path)
            assert tax_hash == files_to_dl[dl_key]["sha256"]
    
    # Untar the spectra data
    tar_path = os.path.join(core_path, "data/lvl0/", "smass2data.tar.gz")
    if os.path.exists(tar_path): 
        # Check if already extracted (simple check if smass2 dir exists)
        if not os.path.exists(os.path.join(core_path, "data/lvl0/", "smass2")):
             print("Extracting tar file...")
             tar = tarfile.open(tar_path, "r:gz")
             tar.extractall(os.path.join(core_path, "data/lvl0/"))
             tar.close()

def parse_data(core_path):
    """
    Parses downloaded files and merges taxonomy classification with spectra data.
    Based on logic from 2_data_parse.ipynb
    """
    print("Step 2: Data Parsing")
    
    # Get a sorted list of all spectra files
    spectra_filepaths = sorted(glob.glob(os.path.join(core_path, "data/lvl0/", "smass2/*spfit*")))
    
    # Separate the filepaths into designation and non-designation files
    des_file_paths = spectra_filepaths[:-8]
    non_file_paths = spectra_filepaths[-8:]
    
    des_file_paths_df = pd.DataFrame(des_file_paths, columns=["FilePath"])
    non_file_paths_df = pd.DataFrame(non_file_paths, columns=["FilePath"])
    
    # Add designation number
    des_file_paths_df.loc[:, "DesNr"] = des_file_paths_df["FilePath"] \
                                            .apply(lambda x: int(re.search(r'a(.*).spfit', pathlib.Path(x).name).group(1)))
    non_file_paths_df.loc[:, "DesNr"] = non_file_paths_df["FilePath"] \
                                            .apply(lambda x: re.search(r'au(.*).spfit', pathlib.Path(x).name).group(1))
    
    # Read the classification file
    asteroid_class_df = pd.read_csv(os.path.join(core_path, "data/lvl0/", "Bus.Taxonomy.txt"),
                                    skiprows=21,
                                    sep="\t",
                                    names=["Name", "Tholen_Class", "Bus_Class", "unknown1", "unknown2"])
    
    # Remove white spaces
    asteroid_class_df.loc[:, "Name"] = asteroid_class_df["Name"].apply(lambda x: x.strip()).copy()
    
    # Separate between designated and non-designated asteroid classes
    des_ast_class_df = asteroid_class_df[:1403].copy()
    non_ast_class_df = asteroid_class_df[1403:].copy()
    
    # Split designated names and get designation number
    des_ast_class_df.loc[:, "DesNr"] = des_ast_class_df["Name"].apply(lambda x: int(x.split(" ")[0]))
    
    # Merge with spectra file paths
    des_ast_class_join_df = des_ast_class_df.merge(des_file_paths_df, on="DesNr")
    
    # Logic for non-designated names
    non_ast_class_df.loc[:, "DesNr"] = non_ast_class_df["Name"].apply(lambda x: x.replace(" ", ""))
    non_ast_class_join_df = non_ast_class_df.merge(non_file_paths_df, on="DesNr")
    
    # Merge both datasets
    asteroids_df = pd.concat([des_ast_class_join_df, non_ast_class_join_df], axis=0)
    asteroids_df.reset_index(drop=True, inplace=True)
    
    # Cleanup columns
    asteroids_df.drop(columns=["Tholen_Class", "unknown1", "unknown2"], inplace=True)
    asteroids_df.dropna(subset=["Bus_Class"], inplace=True)
    
    # Read and store the spectra
    asteroids_df.loc[:, "SpectrumDF"] = asteroids_df["FilePath"].apply(lambda x: pd.read_csv(x, sep="\t",
                                                         names=["Wavelength_in_microm", "Reflectance_norm550nm"]))
    
    asteroids_df.reset_index(drop=True, inplace=True)
    asteroids_df.loc[:, "DesNr"] = asteroids_df["DesNr"].astype(str)
    
    # Save level 1 data
    pathlib.Path(os.path.join(core_path, "data/lvl1")).mkdir(parents=True, exist_ok=True)
    asteroids_df.to_pickle(os.path.join(core_path, "data/lvl1/", "asteroids_merged.pkl"), protocol=4)
    print("Level 1 data saved.")
    
    return asteroids_df

def enrich_data(core_path):
    """
    Enriches dataset with Main Group classification.
    Based on logic from 3_data_enrichment.ipynb
    """
    print("Step 3: Data Enrichment")
    
    # Check if Lvl1 exists, run parse if not
    lvl1_path = os.path.join(core_path, "data/lvl1/", "asteroids_merged.pkl")
    if not os.path.exists(lvl1_path):
        asteroids_df = parse_data(core_path)
    else:
        asteroids_df = pd.read_pickle(lvl1_path)
        
    bus_to_main_dict = {
        'A': 'Other', 'B': 'C', 'C': 'C', 'Cb': 'C', 'Cg': 'C', 'Cgh': 'C', 'Ch': 'C',
        'D': 'Other', 'K': 'Other', 'L': 'Other', 'Ld': 'Other', 'O': 'Other', 'R': 'Other',
        'S': 'S', 'Sa': 'S', 'Sk': 'S', 'Sl': 'S', 'Sq': 'S', 'Sr': 'S',
        'T': 'Other', 'V': 'Other', 'X': 'X', 'Xc': 'X', 'Xe': 'X', 'Xk': 'X'
    }
    
    asteroids_df.loc[:, "Main_Group"] = asteroids_df["Bus_Class"].apply(lambda x: bus_to_main_dict.get(x, "None"))
    
    # Drop columns
    if "DesNr" in asteroids_df.columns:
        asteroids_df.drop(columns=["DesNr"], inplace=True)
    if "FilePath" in asteroids_df.columns:
        asteroids_df.drop(columns=["FilePath"], inplace=True)
        
    # Create Level 2 directory and save
    pathlib.Path(os.path.join(core_path, "data/lvl2")).mkdir(parents=True, exist_ok=True)
    asteroids_df.to_pickle(os.path.join(core_path, "data/lvl2/", "asteroids.pkl"), protocol=4)
    print("Level 2 data (Final Dataset) saved.")
    
    return asteroids_df

def load_data(core_path):
    """
    Loads distinct level data, creates if missing.
    """
    lvl2_path = os.path.join(core_path, "data/lvl2/", "asteroids.pkl")
    
    if os.path.exists(lvl2_path):
        print(f"Loading existing Lvl2 data from {lvl2_path}")
        return pd.read_pickle(lvl2_path)
    else:
        # Check Lvl0 (raw download)
        if not os.path.exists(os.path.join(core_path, "data/lvl0/")):
            download_data(core_path)
        
        # Parse and Enrich
        return enrich_data(core_path)
