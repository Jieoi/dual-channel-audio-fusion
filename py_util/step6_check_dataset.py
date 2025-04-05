"""

# step6_check_dataset.py

This script performs thorough checks on the prepared dataset:
- Verifies file counts in train/test directories
- Confirms label match with audio files
- Plots distribution of health status
- Attempts to load a few audio samples for shape inspection
"""

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def list_files(directory, label):
    files_list = os.listdir(directory)
    total_files = len(files_list)
    sample_files = files_list[:10]
    print(f"Total {label} Files: {total_files}")
    print(f"Sample {label} Files:", sample_files)
    return total_files

def plot_distribution(counts, labels, title):
    total = sum(counts)
    percentages = [(count / total) * 100 for count in counts]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts, color=['lightblue', 'orange'])
    plt.xlabel("Label")
    plt.ylabel("Number of Files")
    plt.title(title)

    for bar, percentage in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, 10, f'{percentage:.1f}%',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.show()

def detect_id_column(csv_path):
    df = pd.read_csv(csv_path, nrows=1)
    possible_columns = ["uuid", "id", "filename"]
    for col in possible_columns:
        if col in df.columns:
            print(f"Detected ID column: '{col}'")
            return col
    raise ValueError(f"No known ID column found in {csv_path}. Columns available: {df.columns.tolist()}")

def load_labels(file_path, dataset_name, id_column=None):
    data = pd.read_csv(file_path)
    if id_column is None:
        id_column = detect_id_column(file_path)
    print(f"Unique health_status labels in {dataset_name} set:", data['health_status'].unique())

    label_dict = {f"{id_}.wav": label for id_, label in zip(data[id_column], data['health_status'])}
    print(f"Sample {dataset_name} labels:", list(label_dict.items())[:5])
    print(f"{len(label_dict)} {dataset_name} files labeled correctly.")
    return label_dict, data['health_status'].value_counts().to_dict()

def load_audio_dataset(audio_dir, label_dict):
    data = []
    labels = []
    filenames = []
    skipped_files = 0

    for filename in label_dict:
        file_path = os.path.join(audio_dir, filename)

        if not os.path.isfile(file_path) or os.path.getsize(file_path) < 10:
            print(f"Skipping {file_path} (file missing or too small)")
            skipped_files += 1
            continue

        try:
            audio, sr = librosa.load(file_path, sr=None)
            data.append(audio)
            labels.append(label_dict[filename])
            filenames.append(filename)
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            skipped_files += 1

    print(f"Skipped {skipped_files} files due to errors.")
    return np.array(data, dtype=object), np.array(labels), filenames

def run_check_pipeline(
    train_dir,
    test_dir,
    train_labels_path,
    test_labels_path,
    id_column=None
):
    # Step 1: Count files
    total_train_files = list_files(train_dir, "Train")
    total_test_files = list_files(test_dir, "Test")

    # Step 2: Load labels and check counts
    train_label_dict, train_label_counts = load_labels(train_labels_path, "training", id_column)
    test_label_dict, test_label_counts = load_labels(test_labels_path, "testing", id_column)

    if len(train_label_dict) == total_train_files:
        print("Training labels match the number of training files.")
    else:
        print(f"Mismatch: {len(train_label_dict)} training labels vs. {total_train_files} training files.")

    if len(test_label_dict) == total_test_files:
        print("Testing labels match the number of testing files.")
    else:
        print(f"Mismatch: {len(test_label_dict)} testing labels vs. {total_test_files} testing files.")

    # Step 3: Plot label distributions
    plot_distribution(list(train_label_counts.values()), list(map(str, train_label_counts.keys())), "Training Label Distribution")
    plot_distribution(list(test_label_counts.values()), list(map(str, test_label_counts.keys())), "Testing Label Distribution")

    # Step 4: Try loading files
    X_train, y_train, train_files = load_audio_dataset(train_dir, train_label_dict)
    X_test, y_test, test_files = load_audio_dataset(test_dir, test_label_dict)

    print(f"\nTraining Data: {X_train.shape[0]} samples")
    print(f"Test Data: {X_test.shape[0]} samples")

    # Step 5: Show sample audio shapes
    print("\nSample training shapes:")
    for audio, name in zip(X_train[:5], train_files[:5]):
        print(f"{name}: {audio.shape}")

    print("\nSample test shapes:")
    for audio, name in zip(X_test[:5], test_files[:5]):
        print(f"{name}: {audio.shape}")
