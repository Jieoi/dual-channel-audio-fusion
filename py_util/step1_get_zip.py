'''
# step1_get_zip.py

This script is used in Google Colab to:
1. Mount Google Drive
2. Extract a .zip file from Google Drive into a specified directory
3. Extract Coswara tar.gz split files into proper folders
'''

import os
import shutil
import zipfile
import subprocess
from google.colab import drive

def mount_drive():
    """
    Mounts Google Drive in Colab
    """
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
        print("Google Drive mounted.")
    else:
        print("Google Drive already mounted.")


# ------------------ COUGHVID FUNCTIONS ------------------

def extract_zip_to_directory(zip_path, extract_to):
    """
    Extracts a .zip file from Google Drive to the specified directory.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found at: {zip_path}")

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Extracted files to: {extract_to}")


# ------------------ COSWARA FUNCTIONS ------------------

def extract_coswara_archives(coswara_data_dir):
    """
    Extracts .tar.gz.* split files inside Coswara dataset folders into a common Extracted_data folder.
    """
    extracted_data_dir = os.path.join(coswara_data_dir, 'Extracted_data')
    os.makedirs(extracted_data_dir, exist_ok=True)

    for subdir in os.listdir(coswara_data_dir):
        sub_path = os.path.join(coswara_data_dir, subdir)
        if os.path.isdir(sub_path) and subdir.startswith("202"):
            part_files = [f for f in os.listdir(sub_path) if f.endswith(('.tar.gz.aa', '.tar.gz.ab', '.tar.gz.ac', '.tar.gz.ad'))]
            if part_files:
                tar_command = f"cat {sub_path}/*.tar.gz.* | tar -xvz -C {extracted_data_dir}/"
                print(f"Extracting: {subdir}")
                subprocess.run(tar_command, shell=True)

    print("Coswara extraction complete.")


def consolidate_coswara_heavy_cough_files(extracted_data_dir, output_dir):
    """
    From Coswara Extracted_data structure, finds and copies all '*_cough-heavy.wav'
    files into a flat output directory with renamed filenames (removes '_cough-heavy').

    Args:
        extracted_data_dir (str): Path to Coswara Extracted_data folder with nested structure.
        output_dir (str): Destination folder for heavy cough files (flattened).
    """
    os.makedirs(output_dir, exist_ok=True)
    copied = 0

    for root, dirs, files in os.walk(extracted_data_dir):
        for file in files:
            if file.endswith("_cough-heavy.wav"):
                src = os.path.join(root, file)

                # Remove "_cough-heavy" from filename
                file_id = file.replace("_cough-heavy.wav", "")
                dst = os.path.join(output_dir, f"{file_id}.wav")

                # Check for overwrites
                if os.path.exists(dst):
                    print(f"Skipped duplicate: {dst}")
                    continue

                shutil.copy2(src, dst)
                copied += 1
                print(f"Copied and renamed: {file} → {file_id}.wav")

    print(f"\nFinished copying. Total heavy cough files extracted and renamed: {copied}")
