import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import get_window
import librosa

def segment_audio(audio, sample_rate, frame_size_ms, overlap):
    frame_size = int(sample_rate * frame_size_ms / 1000)
    hop_size = int(frame_size * (1 - overlap))

    n_frames = int(np.ceil((len(audio) - frame_size) / hop_size)) + 1
    frames = np.zeros((n_frames, frame_size))

    for i in range(n_frames):
        start_idx = i * hop_size
        end_idx = start_idx + frame_size

        if end_idx > len(audio):
            frames[i, :len(audio) - start_idx] = audio[start_idx:]
        else:
            frames[i] = audio[start_idx:end_idx]

    print(f"\nNumber of frames = {len(frames)}\n")
    return frames

def extract_dft_features(frames, n_dft, window_type):
    window = get_window(window_type, frames.shape[1])
    frames_windowed = frames * window

    dft_features = np.abs(np.fft.rfft(frames_windowed, n=n_dft))

    print(f"\nNumber of DFT features = {len(dft_features)}\n")
    return dft_features

def select_dft_features(dft_features, sample_rate, n_dft, freq_range):
    low_freq, high_freq = freq_range

    freqs = np.fft.rfftfreq(n_dft, d=1.0 / sample_rate)

    idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
    selected_features = dft_features[:, idx]

    print(f"\nNumber of selected DFT features = {len(selected_features)}\n")
    return selected_features, freqs[idx]

if __name__ == "__main__":
    # Hyperparameters: --->
    audio_file = "ASS/audios/12_1_mono.wav"
    sample_rate_target = 22050
    frame_size_ms = 10
    overlap = 0.5
    n_dft = 256
    window_type = "hamming"
    freq_range = (100, 8000)

    # Load audio: --->
    sample_rate, audio = wavfile.read(audio_file)

    # Resample if necessary: --->
    if sample_rate != sample_rate_target:
        audio = librosa.resample(audio.astype(np.float32) / np.iinfo(np.int16).max, orig_sr=sample_rate, target_sr=sample_rate_target)
        sample_rate = sample_rate_target
    else:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    # Segment audio into frames: --->
    frames = segment_audio(audio, sample_rate, frame_size_ms, overlap)

    # Extract DFT features: --->
    dft_features = extract_dft_features(frames, n_dft, window_type)

    # Select DFT features: --->
    selected_features, selected_freqs = select_dft_features(dft_features, sample_rate, n_dft, freq_range)

    plt.figure(figsize=(12, 6))

    # Display selected DFT features: --->
    plt.subplot(1, 2, 1)
    plt.imshow(selected_features.T, origin="lower", aspect="auto", cmap="inferno")
    plt.xlabel("Frame Index")
    plt.ylabel("Frequency Bin Index")
    plt.title("Selected DFT Features")

    # Plot frequency response: --->
    plt.subplot(1, 2, 2)
    plt.plot(selected_freqs, selected_features[0])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Response of First Frame")

    plt.tight_layout()
    plt.show()
