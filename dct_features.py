# Import libraries: --->
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules: --->
from utils import compute_dct

# Import constants: --->
from const import SAMPLE_RATE

def calculate_energy_retention(dct_coefficients):
    energy = dct_coefficients ** 2
    total_energy = torch.sum(energy, dim = -1, keepdim = True)

    energy_retention = energy / total_energy
    cumulative_energy_retention = torch.cumsum(energy_retention, dim = -1)

    return energy_retention, cumulative_energy_retention

def frame_audio(audio, frame_length, hop_length):
    num_frames = int(np.ceil((len(audio) - frame_length) / hop_length)) + 1
    frames = torch.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_length
        if end_idx > len(audio):
            frames[i, :len(audio) - start_idx] = torch.from_numpy(audio[start_idx:len(audio)])
        else:
            frames[i] = torch.from_numpy(audio[start_idx:end_idx])

    print(f"\nNumber of frames = {len(frames)}\n")

    return frames

if __name__ == "__main__":
    # Hyperparameters: --->
    frame_size_ms = 10
    hop_size_ms = 5
    frame_length = int(frame_size_ms * SAMPLE_RATE / 1000)
    hop_length = int(hop_size_ms * SAMPLE_RATE / 1000)
    wav_file = "ASS/audios/21_1_mono.wav"

    # Load audio file: --->
    audio, sr = librosa.load(wav_file, sr = SAMPLE_RATE)

    # Resample if necessary: --->
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr = sr, target_sr = SAMPLE_RATE)

    # Calculate DCT coefficients: --->
    frames = frame_audio(audio, frame_length, hop_length)
    dct_coefficients = torch.tensor(compute_dct(frames.numpy()), dtype=torch.float32)                          # Return only the magnitude (Real value), exclude the phase (complex value)

    print(f"Number of DCT coefficients = {len(dct_coefficients[0])}\n")

    # Calculate energy retention: --->
    energy_retention, cumulative_energy_retention = calculate_energy_retention(dct_coefficients)

    # Plot DCT coefficients for the first frame: --->
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(dct_coefficients[0].numpy())
    plt.title("DCT Coefficients (First Frame)")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")

    # Plot coefficient-wise energy retention for the first frame: --->
    plt.subplot(1, 3, 2)
    plt.plot(energy_retention[0].numpy())
    plt.title("Coefficient-wise Energy Retention (First Frame)")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Energy Retention")

    # Plot cumulative energy retention for the first frame: --->
    plt.subplot(1, 3, 3)
    plt.plot(cumulative_energy_retention[0].numpy())
    plt.title("Cumulative Energy Retention (First Frame)")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Cumulative Energy Retention")
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Energy Retention')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Energy Retention')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Find the number of coefficients that capture 90% and 95% of the energy for each frame: --->
    num_coefficients = (cumulative_energy_retention >= 0.99).float().argmax(dim = -1) + 1           # Change the threshold if needed (0.99 = 99%)
    print(f"Number of coefficients that capture 99% energy: {torch.max(num_coefficients)}")
