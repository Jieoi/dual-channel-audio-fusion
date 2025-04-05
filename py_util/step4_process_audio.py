"""

# step4_process_audio.py

Audio preprocessing pipeline for datasets like COUGHVID, COSWARA, or LDM-generated audio.

Includes:
- Audio loading
- SNR-based filtering
- Resampling
- Noise reduction
- Silence trimming
- Saving processed .wav files

Designed to work with flat directory structures.
"""

import os
import numpy as np

# -------------------------
# Dependency Checks
# -------------------------
import importlib
missing = []
for pkg in ["librosa", "soundfile", "noisereduce"]:
    if importlib.util.find_spec(pkg) is None:
        missing.append(pkg)

if missing:
    print(f"\nMissing required packages: {', '.join(missing)}")
    print(f"Please install them by running: !pip install {' '.join(missing)}")
    print("After installing, you may need to restart the runtime.")
    raise ImportError("One or more required packages are missing.")

import librosa
import soundfile as sf
import noisereduce as nr

def load_audio(file_path, min_duration=0.1):
    """
    Loads an audio file and filters out very short clips (duration < min_duration).
    Returns:
        audio: np.array or None
        sr: int or None
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        duration = len(audio) / sr
        if duration < min_duration:
            return None, None
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}")
        return None, None

def calculate_snr(audio):
    """
    Calculates Signal-to-Noise Ratio (SNR) using 90th percentile energy threshold.
    Returns 0 if no cough segments are detected.
    """
    audio = audio / np.max(np.abs(audio))
    energy = np.abs(audio)
    threshold = np.percentile(energy, 90)
    cough_indices = np.where(energy > threshold)[0]

    if len(cough_indices) == 0:
        return 0

    cough_signal = audio[cough_indices]
    noise_signal = np.delete(audio, cough_indices)
    cough_power = np.mean(cough_signal ** 2) if len(cough_signal) > 0 else 1e-10
    noise_power = np.mean(noise_signal ** 2) if len(noise_signal) > 0 else 1e-10

    snr = 20 * np.log10(np.sqrt(cough_power) / np.sqrt(noise_power))
    return snr

def resample_audio(audio, sr_orig, target_sr=12000):
    """
    Resamples audio to target sample rate if original is higher.
    """
    if sr_orig > target_sr:
        return librosa.resample(audio, orig_sr=sr_orig, target_sr=target_sr)
    return audio

def denoise_audio(audio, sr):
    """
    Applies non-stationary noise reduction algorithm.
    """
    return nr.reduce_noise(y=audio, sr=sr, stationary=False)

def trim_silence(audio, top_db=25):
    """
    Trims silence from beginning and end using top_db decibel threshold.
    """
    return librosa.effects.trim(audio, top_db=top_db)[0]

def process_audio(file_path):
    """
    Processes a single file:
    - Load
    - Filter low-SNR
    - Resample
    - Denoise
    - Trim silence
    Returns cleaned audio or None if invalid.
    """
    audio, sr = load_audio(file_path)
    if audio is None:
        return None

    snr = calculate_snr(audio)
    if snr < 10:
        print(f"Skipping {file_path} due to low SNR value at {snr:.2f} dB")
        return None

    audio = resample_audio(audio, sr, target_sr=12000)
    audio = denoise_audio(audio, sr)
    audio = trim_silence(audio)
    return audio

def process_and_save_dataset(input_folder, output_folder):
    """
    Processes all .wav files in the input folder and saves cleaned versions to output folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    audio_files = os.listdir(input_folder)
    total_files = len(audio_files)
    processed_count = 0
    saved_count = 0
    not_processed_count = 0

    for file_name in audio_files:
        file_path = os.path.join(input_folder, file_name)
        processed_segment = process_audio(file_path)
        processed_count += 1

        if processed_segment is None or len(processed_segment) == 0:
            print(f"Skipping {file_name}: No valid audio after processing.")
            not_processed_count += 1
            continue

        output_path = os.path.join(output_folder, file_name)
        sf.write(output_path, processed_segment, 12000)
        saved_count += 1
        print(f"Saved: {output_path}")

    print("\nSummary of Processing:")
    print(f"- Total number of files in dataset: {total_files}")
    print(f"- Files attempted to process: {processed_count}")
    print(f"- Files successfully saved: {saved_count}")
    print(f"- Files not processed: {not_processed_count}")