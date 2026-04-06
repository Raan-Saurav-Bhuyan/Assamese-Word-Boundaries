import numpy as np
import librosa

# Import constants: --->
from const import FRAME_SIZE, FRAME_STRIDE, SAMPLE_RATE, NUM_FFT, NUM_MFCC, MIN_WORD_DURATION, AMPLITUDE_THRESHOLD, ENERGY_THRESHOLD, INPUT_DIM

# Audio Segmentation into overlapping frames: --->
def segment_audio(audio):
    frame_len = int(FRAME_SIZE * SAMPLE_RATE)
    frame_step = int(FRAME_STRIDE * SAMPLE_RATE)
    signal_len = len(audio)
    num_frames = int(np.ceil((signal_len - frame_len) / frame_step)) + 1
    pad_len = num_frames * frame_step + frame_len

    pad_signal = np.append(audio, np.zeros(pad_len - signal_len))

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T

    frames = pad_signal[indices.astype(np.int32)]       #shape: (num_frames, frame_len)

    return frames

# Extract the Discrete Fourier Transform (DFT) features: --->
def compute_dft(frames, n_fft = NUM_FFT):
    n_fft = n_fft or frames.shape[-1]
    dft = np.fft.fft(frames, n=n_fft)
    dft_magnitude = np.abs(dft[:, :n_fft // 2])

    return dft_magnitude.astype(np.float32)         # shape: (num_frames, n_fft//2)

def compute_mfcc(frames):
    mfcc_features = []

    win_length = int(SAMPLE_RATE * FRAME_SIZE)
    hop_length = int(SAMPLE_RATE * FRAME_STRIDE)

    # Extract the MFCC features for each of the frames: --->
    for frame in frames:
        mfcc = librosa.feature.mfcc(
            y = frame,
            sr = SAMPLE_RATE,
            n_mfcc = NUM_MFCC,
            n_fft = NUM_FFT,
            win_length = win_length,
            hop_length = hop_length,
            center = False                                                  # prevent librosa from adding padding
        )

        # Since frame is a single chunk, take just one MFCC vector: --->
        mfcc_features.append(mfcc[:, 0])                        # Extract one feature vector per frame, shape: (n_mfcc,)

    return np.array(mfcc_features, dtype=np.float32)  # shape: (num_frames, n_mfcc)

def compute_dct(frames):
    # Reflect the input for even-symmetry extension (DCT-II): --->
    mirrored = np.concatenate([frames, frames[..., -2: 0: -1]], axis = -1)

    # Zero-pad if needed to match NUM_FFT: --->
    if mirrored.shape[-1] < NUM_FFT:
        mirrored = np.pad(mirrored, ((0, 0), (0, NUM_FFT - mirrored.shape[-1])), mode = 'constant')

    # FFT computation: --->
    X = np.fft.fft(mirrored, n=NUM_FFT)

    # DCT-II scaling factor: --->
    k = np.arange(INPUT_DIM)
    scale = 2 * np.exp(-1j * np.pi * k / (2 * NUM_FFT))

    # Compute real part and apply scaling: --->
    dct_features = (scale * X[:, :INPUT_DIM]).real

    # Manual orthonormalization: --->
    dct_features *= np.sqrt(2 / NUM_FFT)
    dct_features[:, 0] /= np.sqrt(2)

    return dct_features.astype(np.float32)                # Shape: (num_frames, dct_features)

# Read .align file (sample-level): --->
def read_alignment(filepath):
    starts_ends = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 3:
                continue

            start_sample = int(parts[0])
            end_sample = int(parts[1])

            label = parts[2]

            if label not in ['sil', 'sp']:  # skip silence
                starts_ends.append((start_sample / SAMPLE_RATE, end_sample / SAMPLE_RATE, label))

    return starts_ends

def estimate_amplitude_threshold(audio):
    # audio_abs = np.abs(audio)
    hist, bin_edges = np.histogram(audio, bins = 100, density = True)

    # Find the "knee" point where energy starts to rise sharply
    energy_diff = np.diff(hist)
    knee_idx = np.argmax(energy_diff > 2e-3)  # You can tweak this threshold

    threshold = bin_edges[knee_idx]

    # if plot:
    #     plt.figure(figsize=(8, 4))
    #     plt.hist(audio_abs, bins=100, color='gray', alpha=0.7, label='Amplitude Histogram')
    #     plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ≈ {threshold:.3f}')
    #     plt.title('Estimated Amplitude Threshold')
    #     plt.xlabel('Amplitude')
    #     plt.ylabel('Density')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    return threshold

def mad_based_amplitude_threshold(audio, k = 3.5):
    median = np.median(audio)
    mad = np.median(np.abs(audio - median))
    threshold = median + k * mad

    # if plot:
    #     plt.figure(figsize=(8, 4))
    #     plt.hist(audio_abs, bins=100, color='gray', alpha=0.7)
    #     plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ≈ {threshold:.3f}')
    #     plt.title("MAD-Based Amplitude Thresholding")
    #     plt.xlabel("Amplitude")
    #     plt.ylabel("Count")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    return threshold

def amplitude_threshold(audio):
    # Load the audio waveform: --->
    # audio, _ = librosa.load(audio_path, sr = SAMPLE_RATE)

    # Load the audio samples as an numpy array: --->
    audio = np.abs(audio)

    # AMPLITUDE_THRESHOLD = estimate_amplitude_threshold(audio)
    # print(f"\nThreshold = {AMPLITUDE_THRESHOLD}\n")

    # print(f"\nAudio as numpy array:\n{max(audio)}\n")

    signal_len = len(audio)
    starts_ends = []

    # Create a boolean mask where each sample in the audio signal is compared against the threshold: --->
    speech = audio > AMPLITUDE_THRESHOLD

    # convert the boolean array to integers and compute the difference between adjacent values: --->
    changes = np.diff(speech.astype(int))

    # start_indices record samples where transition from silence to speech occurred;
    # end_indices record samples where transition from speech to silence occurred: --->
    start_indices = np.where(changes == 1)[0]       # 0 -> 1 transition (i.e. False -> True); becomes +1, start of speech
    end_indices = np.where(changes == -1)[0]       # 1 -> 0 transition (i.e. True -> False); becomes -1, end of speech

    # Handle edge cases (start or end of signal is speech): --->
    if len(start_indices) == 0 and len(end_indices) == 0:
        return []

    # Ensure the first speech segment has a valid end: --->
    # Case 1: No -1 transitions were detected - meaning end of speech.
    # Case 2: First detected start (+1 transition) comes after the first detected end (-1 transition).
    if len(end_indices) == 0 or (len(start_indices) > 0 and start_indices[0] > end_indices[0]):
        end_indices = np.insert(end_indices, 0, 0)

    # Ensure the last speech segment is properly closed: --->
    # Case 1: No +1 transitions were detected - meaning no start of speech.
    # Case 2: Last detected start (+1 transition) comes after the last detected end (-1 transition).
    if len(start_indices) == 0 or start_indices[-1] > end_indices[-1]:
        end_indices = np.append(end_indices, signal_len - 1)

     # Convert samples to time (in seconds): --->
    for start, end in zip(start_indices, end_indices):
        start_time = start / SAMPLE_RATE
        end_time = end / SAMPLE_RATE

        if end_time - start_time >= MIN_WORD_DURATION:
            starts_ends.append((start_time, end_time))

    return starts_ends

def energy_threshold(audio, frames = None):
    # Load audio waveform: --->
    # audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    starts_ends = []

    # Segment the audio into overlapping frames: --->
    if frames is None:
        frames = segment_audio(audio)

    # Compute energy per frame: --->
    energy = np.sum(frames ** 2, axis = 1)

    # Create a boolean mask where each frame in the audio signal is compared against the threshold: --->
    speech = energy > ENERGY_THRESHOLD

    # convert the boolean array to integers and compute the difference between adjacent values: --->
    changes = np.diff(speech.astype(int))

    # start_indices record samples where transition from silence to speech occurred;
    # end_indices record samples where transition from speech to silence occurred: --->
    start_indices = np.where(changes == 1)[0]           # 0 -> 1 transition (i.e. False -> True); becomes +1,start of speech
    end_indices = np.where(changes == -1)[0]           # 1 -> 0 transition (i.e. True -> False); becomes -1, end of speech

    # Handle edge cases (start or end of signal is speech): --->
    if len(start_indices) == 0 and len(end_indices) == 0:
        return []

    # Ensure the first speech segment has a valid end: --->
    # Case 1: No -1 transitions were detected - meaning end of speech.
    # Case 2: First detected start (+1 transition) comes after the first detected end (-1 transition).
    if len(end_indices) == 0 or (len(start_indices) > 0 and start_indices[0] > end_indices[0]):
        end_indices = np.insert(end_indices, 0, 0)

    # Ensure the last speech segment is properly closed: --->
    # Case 1: No +1 transitions were detected - meaning no start of speech.
    # Case 2: Last detected start (+1 transition) comes after the last detected end (-1 transition).
    if len(start_indices) == 0 or start_indices[-1] > end_indices[-1]:
        end_indices = np.append(end_indices, len(energy) - 1)

    # Convert frame indices to time (in seconds): --->
    for start, end in zip(start_indices, end_indices):
        start_time = start * FRAME_STRIDE
        end_time = end * FRAME_STRIDE + FRAME_SIZE

        if end_time - start_time >= MIN_WORD_DURATION:
            starts_ends.append((start_time, end_time))

    return starts_ends
