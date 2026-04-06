# Import libraries: --->
import torch as tc
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules: --->
from utils import read_alignment, segment_audio, compute_dft, compute_mfcc, compute_dct, amplitude_threshold, energy_threshold
# from model import MLP
from model import BiRNN

# Import constants: --->
from const import SAMPLE_RATE, MODEL_NAME, FRAME_SIZE, FRAME_STRIDE, THRESHOLD, FEATURES, CONV, MODEL

def plot_waveform_gt(audio_path, align_path):
    waveform, sr = librosa.load(audio_path, sr = SAMPLE_RATE)

    # Load the ground truth boundaries: --->
    if THRESHOLD == "N":
        boundaries = read_alignment(align_path)
    elif THRESHOLD == "A":
        boundaries = amplitude_threshold(waveform)
    elif THRESHOLD == "E":
        boundaries = energy_threshold(waveform)
    else:
        raise ValueError("\nThe type of ground truth is not properly defined!\nExiting...\n")

    # print(f"\nBoundaries = {boundaries}\n")

    plt.figure(figsize=(14, 4))
    plt.title("Sample Audio in Waveform with Energy Thresholding GT Word Boundaries")

    librosa.display.waveshow(waveform, sr = sr, alpha = 0.6)

    for start, end in boundaries:
        plt.axvline(x = start, color = 'red', linestyle = '-', alpha = 0.7)
        plt.axvline(x = end, color='red', linestyle = '-', alpha = 0.7)
        # plt.text((start + end) / 2, 0.6 * max(waveform), word, rotation=45, fontsize=9, ha='center')

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.xlim([0, 3.0])
    plt.tight_layout()
    plt.show()

def plot_waveform_predicted(audio_path, align_path):
    # Load the audio waveform: --->
    waveform, sr = librosa.load(audio_path, sr = SAMPLE_RATE)

    # Segmentation of audio, extraction of frame-wise features: --->
    frames = segment_audio(waveform)

    if FEATURES == "DFT":
        features = compute_dft(frames)              #shape: (seq_len, input_dim)
    elif FEATURES == "MFCC":
        features = compute_mfcc(frames)
    elif FEATURES == "DCT":
        features = compute_dct(frames)
    else:
        raise TypeError("\nThe type of features is not properly defined!\nExiting...\n")

    features_tensor = tc.tensor(features, dtype = tc.float32)

    # (Case Conv1D): Since the shape of the features is (seq_len, input_dim), make it (1, seq_len, input_dim): --->
    if CONV and features_tensor.dim() == 2:
        features_tensor = features_tensor.unsqueeze(0)  # Shape: (1, seq_len, input_dim)

    # Load the ground truth boundaries: --->
    if THRESHOLD == "N":
        boundaries = read_alignment(align_path)
    elif THRESHOLD == "A":
        boundaries = amplitude_threshold(waveform)
    elif THRESHOLD == "E":
        boundaries = energy_threshold(waveform, frames)
    else:
        raise ValueError("\nThe type of ground truth is not properly defined!\nExiting...\n")

    # Load the model to predict the boundaries: --->
    model = BiRNN(MODEL)
    model.load_state_dict(tc.load(MODEL_NAME))

    # Predict boundaries: --->
    model.eval()
    with tc.no_grad():
        outputs = model(features_tensor)
        probs = tc.sigmoid(outputs).squeeze().numpy()
        pred_indices = np.where(probs > 0.5)[0]

    # Convert prediction indices to time: --->
    pred_times = pred_indices * FRAME_STRIDE + FRAME_SIZE / 2

    # Plot size: --->
    plt.figure(figsize = (14, 4))
    librosa.display.waveshow(waveform, sr = sr, alpha = 0.6)
    plt.title("Word Boundaries: Ground Truth (Weakly Labeled) vs Bi-LSTM (DFT Trained) Prediction")

    # Plot ground truth boundaries: --->
    for (start, end, word) in boundaries:
        plt.axvline(x = start, color = 'red', linestyle = '-', alpha = 0.8, label='Ground Truth Start' if start == boundaries[0][0] else "")
        plt.axvline(x = end, color = 'red', linestyle = '-', alpha = 0.8, label='Ground Truth Start' if start == boundaries[0][1] else "")
        plt.text((start + end) / 2, 0.6 * max(waveform), word, rotation = 45, fontsize = 9, ha = 'center')

    # Plot predicted boundaries: --->
    for t in pred_times:
        plt.axvline(x = t, color = 'black', linestyle = '--', alpha = 0.6, label=pred_times[t])

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.xlim([0, 3.0])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Plot frame-wise word occurrence probability with ground truth overlay: --->
def plot_probability_profile(audio_path, align_path):
    waveform, sr = librosa.load(audio_path, sr = SAMPLE_RATE)

    # Load the ground truth boundaries: --->
    if THRESHOLD == "N":
        boundaries = read_alignment(align_path)
    elif THRESHOLD == "A":
        boundaries = amplitude_threshold(waveform)
    elif THRESHOLD == "E":
        boundaries = energy_threshold(waveform, frames)
    else:
        print("\nThe type of ground truth is not properly defined!\nExiting...\n")
        quit()

    # Segment audio into frames and extract features: --->
    frames = segment_audio(waveform)

    if FEATURES == "DFT":
        features = compute_dft(frames)              #shape: (seq_len, input_dim)
    elif FEATURES == "MFCC":
        features = compute_mfcc(frames)
    else:
        raise ValueError("\nThe type of features is not properly defined!\nExiting...\n")

    feature_tensor = tc.tensor(features, dtype = tc.float32)

    # Load the model to predict the boundaries: --->
    model = BiRNN(MODEL)
    model.load_state_dict(tc.load(MODEL_NAME))

    # Model inference: --->
    model.eval()
    with tc.no_grad():
        outputs = model(feature_tensor).squeeze()
        probs = tc.sigmoid(outputs).numpy()

    # Plot probabilities with frame indices: --->
    frame_indices = np.arange(len(probs))

    plt.figure(figsize=(12, 4))
    plt.plot(frame_indices, probs, label="Predicted Probability", color='blue', linewidth=1.2)
    # plt.axhline(y=threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")

    # Overlay shaded ground-truth regions using frame indices: --->
    for start, end in boundaries:
        start_frame = int(start / FRAME_STRIDE)
        end_frame = int(end / FRAME_STRIDE)
        plt.axvspan(start_frame, end_frame, color='sandybrown', alpha=0.3)

    plt.xlabel("Frame Index")
    plt.ylabel("Probability")
    plt.title(f"Word Occurrence Probability (Frame-wise) - {audio_path.split('/')[-1]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    wav_file = "ASS/audios/78_3_mono.wav"
    align_file = "ASS/textgrid/78_3_mono.align"

    # plot_waveform_gt(wav_file, align_file)
    plot_waveform_predicted(wav_file, align_file)
    # plot_probability_profile(wav_file, align_file)
