"""

# step5_split_and_label.py 

Generic dataset splitting and labeling script for any audio dataset:
- Loads processed .wav files
- Matches to original labels using ID column (uuid, id, or filename)
- Saves filtered label CSV
- Splits into train/test sets (stratified)
- Copies .wav files to appropriate folders
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


def detect_id_column(csv_path):
    """
    Auto-detects the ID column in a label CSV.
    Priority order: uuid → id → filename

    Returns:
        str: the detected ID column name
    """
    df = pd.read_csv(csv_path, nrows=1)
    possible_columns = ["uuid", "id", "filename"]

    for col in possible_columns:
        if col in df.columns:
            print(f"Detected ID column: '{col}'")
            return col

    raise ValueError(f"No known ID column found in {csv_path}. Columns available: {df.columns.tolist()}")


def generate_labels_csv(processed_audio_dir, original_labels_csv, output_labels_csv, id_column=None):
    """
    Matches .wav files in processed folder to entries in label CSV using the given ID column.
    Outputs a filtered CSV with only matched entries.
    """
    processed_files = os.listdir(processed_audio_dir)
    processed_ids = [os.path.splitext(file)[0] for file in processed_files if file.endswith(".wav")]
    print(f"Found {len(processed_ids)} processed audio files")

    labels = pd.read_csv(original_labels_csv)

    if id_column is None:
        id_column = detect_id_column(original_labels_csv)

    if id_column not in labels.columns:
        raise ValueError(f"'{id_column}' column not found in label file. Available columns: {labels.columns.tolist()}")

    labels[id_column] = labels[id_column].astype(str).apply(lambda x: os.path.splitext(x)[0])
    matched_labels = labels[labels[id_column].isin(processed_ids)]

    if len(matched_labels) == 0:
        print("No matching audio files found for labels. Aborting CSV save.")
        return pd.DataFrame()

    matched_labels.to_csv(output_labels_csv, index=False)

    healthy = (matched_labels['health_status'] == 1).sum()
    not_healthy = (matched_labels['health_status'] == 0).sum()

    print(f"\nSummary of Label Distribution:")
    print(f"- Matched Files: {len(matched_labels)}")
    print(f"- Healthy: {healthy}")
    print(f"- Not Healthy: {not_healthy}")
    print(f"- New labels CSV saved to: {output_labels_csv}")

    return matched_labels


def split_and_save_labels(matched_labels, output_train_csv, output_test_csv, test_size=0.2, random_seed=42):
    """
    Splits matched label DataFrame into train/test CSVs using stratified sampling on health_status.
    """
    train_labels, test_labels = train_test_split(
        matched_labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=matched_labels['health_status']
    )

    train_labels.to_csv(output_train_csv, index=False)
    test_labels.to_csv(output_test_csv, index=False)

    print(f"\nTrain-test split complete:")
    print(f"- Train labels saved to: {output_train_csv} ({len(train_labels)} rows)")
    print(f"- Test labels saved to: {output_test_csv} ({len(test_labels)} rows)")

    return train_labels, test_labels


def copy_audio_files(label_df, source_folder, destination_folder, id_column='uuid'):
    """
    Copies .wav files listed in label_df from source_folder to destination_folder using IDs in the given column.
    """
    os.makedirs(destination_folder, exist_ok=True)
    copied = 0

    for file_id in label_df[id_column]:
        file_name = f"{file_id}.wav"
        src = os.path.join(source_folder, file_name)
        dest = os.path.join(destination_folder, file_name)

        if os.path.exists(src):
            try:
                shutil.copy(src, dest)
                copied += 1
            except OSError as e:
                print(f"Copy failed for {src} → {dest}. Reason: {e}")
        else:
            print(f"Warning: {src} not found.")

    print(f"Copied {copied}/{len(label_df)} files to {destination_folder}")


def full_pipeline(
    processed_audio_dir,
    original_labels_csv,
    output_labels_csv,
    train_folder,
    test_folder,
    train_csv_path,
    test_csv_path,
    id_column=None
):
    """
    Complete labeling and splitting pipeline:
    - Match processed audio to labels
    - Save matched labels
    - Split into train/test
    - Copy files to corresponding folders

    If id_column is not provided, it is automatically detected.
    """
    matched_labels = generate_labels_csv(processed_audio_dir, original_labels_csv, output_labels_csv, id_column)

    if matched_labels.empty:
        print("No labels matched. Aborting train-test split and copy.")
        return

    if id_column is None:
        id_column = detect_id_column(original_labels_csv)

    train_labels, test_labels = split_and_save_labels(matched_labels, train_csv_path, test_csv_path)

    copy_audio_files(train_labels, processed_audio_dir, train_folder, id_column)
    copy_audio_files(test_labels, processed_audio_dir, test_folder, id_column)

    print("\nLabeling and train/test split complete.")
    print(f"Train audio files saved to: {train_folder}")
    print(f"Test audio files saved to: {test_folder}")