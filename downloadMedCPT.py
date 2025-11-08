#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from pathlib import Path
import sys
import os

# Disable a noisy warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Define the model name and where to save it
MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
SAVE_PATH = Path(__file__).resolve().parent / "local-medcpt-model"

if __name__ == "__main__":
    print(f"--- Attempting to download model: {MODEL_NAME} ---")

    # ---
    # Step 1: Download files using huggingface_hub and get the local path
    # ---
    try:
        print("\nStep 1/3: Downloading files via snapshot_download...")
        # This downloads all files and returns the path to the snapshot directory
        snapshot_path = snapshot_download(repo_id=MODEL_NAME)
        print(f"...Files downloaded to cache at: {snapshot_path}")
    except Exception as e:
        print(f"\n--- CORE DOWNLOAD FAILED ---")
        print(f"Failed to download model snapshot. Are you connected to the internet?")
        print(f"Error details: {e}")
        print("----------------------------")
        sys.exit(1)

    # ---
    # Step 2: Initialize SentenceTransformer FROM THE LOCAL PATH
    # This bypasses the name lookup bug.
    # ---
    try:
        print("\nStep 2/3: Initializing SentenceTransformer from local snapshot...")
        # We pass the *local directory path* to the constructor
        st_model = SentenceTransformer(snapshot_path)
        print("...SentenceTransformer initialized successfully.")
    except Exception as e:
        print(f"\n--- SENTENCE-TRANSFORMER FAILED ---")
        print(f"Failed to load model from path: {snapshot_path}")
        print(f"Error details: {e}")
        print("-----------------------------------")
        sys.exit(1)
        
    # ---
    # Step 3: Save the model in the correct local format.
    # ---
    print(f"\nStep 3/3: Saving model to final local directory: {SAVE_PATH}")
    st_model.save(str(SAVE_PATH))
    
    print("\n--- All Done ---")
    print(f"Model saved successfully to: {SAVE_PATH}")
    print("Your 'search.py' script is now ready to use.")