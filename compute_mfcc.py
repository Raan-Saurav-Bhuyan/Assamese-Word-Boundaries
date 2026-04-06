import os
import numpy as np
import librosa
from const import SAMPLE_RATE, DATASET_ROOT, DATASET_SAMPLES, MFCC_ROOT

# Import custom libraries: --->
from utils import segment_audio, compute_mfcc

# Main MFCC Preprocessing: --->
def save_extracted_mfcc(audio_dir, output_dir):
    os.makedirs(output_dir, exist_ok = True)

    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith(".wav"):
            continue

        audio_path = os.path.join(audio_dir, fname)
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        frames = segment_audio(audio)
        mfcc = compute_mfcc(frames)

        np.save(os.path.join(output_dir, fname.replace(".wav", ".npy")), mfcc)

# Run preprocessing: --->
if __name__ == "__main__":
    audio_dir = f"{DATASET_ROOT}/{DATASET_SAMPLES}"
    output_dir = f"{DATASET_ROOT}/{MFCC_ROOT}"

    save_extracted_mfcc(audio_dir, output_dir)
