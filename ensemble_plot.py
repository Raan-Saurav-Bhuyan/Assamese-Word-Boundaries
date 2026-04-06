# Import libraries: --->
import os
import torch as tc
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Import custom modules: --->
from utils import segment_audio, compute_dft, compute_mfcc, compute_dct, read_alignment
from ensemble_model import EnsembleModel
from model import BiRNN

# Import constants: --->
from const import SAMPLE_RATE, FRAME_STRIDE, FEATURES
AUDIO_PATH = "ASS/audios/69_1_mono.wav"
ALIGN_PATH = "ASS/textgrid/69_1_mono.align"
DEVICE = "cuda" if tc.cuda.is_available() else "cpu"

@tc.no_grad()
def predict_boundaries(audio_path, ensemble_model, threshold = 0.5):
    waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    frames = segment_audio(waveform)

    # Feature extraction
    if FEATURES == "DFT":
        features = compute_dft(frames)
    elif FEATURES == "DCT":
        features = compute_dct(frames)
    elif FEATURES == "MFCC":
        features = compute_mfcc(frames)
    else:
        raise ValueError("Unsupported feature type.")

    features_tensor = tc.tensor(features, dtype=tc.float32).unsqueeze(0).to(DEVICE)

    # Forward pass
    ensemble_model.eval()
    outputs = ensemble_model(features_tensor)                                               # shape: (1, T)
    preds = (outputs.squeeze(0) > threshold).cpu().numpy().astype(int)

    # Convert prediction frame indices to times
    pred_time_stamps = np.arange(len(preds)) * FRAME_STRIDE
    pred_boundaries = pred_time_stamps[preds == 1]

    return waveform, preds, pred_boundaries

def plot_predictions(waveform, sr, pred_boundaries, gt_boundaries, audio_name):
    duration = len(waveform) / sr
    time_axis = np.linspace(0, duration, num=len(waveform))

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, waveform, label='Waveform')

    for t in pred_boundaries:
        plt.axvline(x=t, color='black', linestyle='--', alpha=0.6, label='Predicted' if t == pred_boundaries[0] else "")

    for (start, end, label) in gt_boundaries:
        plt.axvline(x=start, color='red', linestyle='-', alpha=0.5, label='Ground Truth Start' if start == gt_boundaries[0][0] else "")
        plt.axvline(x=end, color='red', linestyle='-', alpha=0.5, label='Ground Truth End' if end == gt_boundaries[0][1] else "")
        plt.text((start + end) / 2, 0.6 * max(waveform), label, rotation = 45, fontsize = 9, ha = 'center')

    plt.title(f"Predicted vs Ground Truth Word Boundaries: {audio_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

def load_model(path, model):
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    model = BiRNN(model).to(device)

    model.load_state_dict(tc.load(path))

    return model

if __name__ == "__main__":
    # Set the model paths: --->
    model_paths = [
        ("checkpoints/wbd_ass_nt_dft_bilstm_1.pth", "BiLSTM"),            # Bi-LSTM + DFT
        # ("checkpoints/wbd_ass_nt_dct_bilstm_1.pth", "BiLSTM"),           # Bi-LSTM + DCT
        ("checkpoints/wbd_ass_nt_dft_bigru_1.pth", "BiGRU")               # Bi-GRU  + DFT
        # ("checkpoints/wbd_ass_nt_dct_bigru_1.pth", "BiGRU")              # Bi-GRU  + DCT
    ]

    # Load the independent models from the defined path: --->
    # models = [load_model(path) for path in model_paths]
    models = []

    for model_path in model_paths:
        models.append(load_model(model_path[0], model_path[1]))

    # Build attention ensemble
    ensemble_model = EnsembleModel(models).to(DEVICE)

    # Predict
    waveform, preds, pred_boundaries = predict_boundaries(AUDIO_PATH, ensemble_model)

    # Ground truth
    gt_boundaries = read_alignment(ALIGN_PATH)

    # Display
    plot_predictions(waveform, SAMPLE_RATE, pred_boundaries, gt_boundaries, os.path.basename(AUDIO_PATH))

    # Print
    print(f"\nPredicted Boundary Times (s):\n{pred_boundaries}")
    print("\nGround Truth Word Segments:")
    for start, end, label in gt_boundaries:
        print(f"{label}: {start:.2f}s - {end:.2f}s")
