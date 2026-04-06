# Import libraries: --->
import os
import torch as tc
import numpy as np
import librosa
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Import custom modules: --->
from utils import segment_audio, compute_dft, compute_mfcc, compute_dct,  read_alignment, amplitude_threshold, energy_threshold

# Import constants: --->
from const import DATASET_SAMPLES, DATASET_LABELS, MFCC_ROOT, FRAME_STRIDE, SAMPLE_RATE, THRESHOLD, FEATURES

class WordBoundaryDataset(Dataset):
    def __init__(self, root_dir):
        # Assign the audio dir, label dir and the list of files in the audio dir: --->
        self.audio_dir = os.path.join(root_dir, DATASET_SAMPLES)
        self.align_dir = os.path.join(root_dir, DATASET_LABELS)
        self.mfcc_dir = os.path.join(root_dir, MFCC_ROOT)                                                  # precomputed MFCC path
        self.paths = [f for f in os.listdir(self.audio_dir) if f.endswith(".wav")]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load the .wav audio samples one by one by the index in the list of files: --->
        wav_filename = self.paths[idx]
        wav_path = os.path.join(self.audio_dir, wav_filename)
        align_path = os.path.join(self.align_dir, wav_filename.replace(".wav", ".align"))

        # waveform, sr = torchaudio.load(wav_path)
        waveform, sr = librosa.load(wav_path, sr = SAMPLE_RATE)

        frames = segment_audio(waveform)

        if FEATURES == "DFT":
            features = compute_dft(frames)              #shape: (seq_len, input_dim)
        elif FEATURES == "MFCC":
            mfcc_path = os.path.join(self.mfcc_dir, wav_filename.replace(".wav", ".npy"))

            if os.path.exists(mfcc_path):
                features = np.load(mfcc_path)
            else:
                # Fallback if precomputed MFCC is not found: --->
                features = compute_mfcc(frames)
        elif FEATURES == "DCT":
            features = compute_dct(frames)
        else:
            raise ValueError("\nThe type of features is not properly defined!\nExiting...\n")

        #! Debugging: --->
        # print(f"\nNumber of frames: {len(frames)}")
        # print(f"\nNumber of features: {len(features)}")

        labels = np.zeros(len(frames))

        #! Debugging: --->
        # print(f"Number of labels: {len(labels)}")

        if THRESHOLD == "N":
            boundaries = read_alignment(align_path)
        elif THRESHOLD == "A":
            boundaries = amplitude_threshold(waveform)
        elif THRESHOLD == "E":
            boundaries = energy_threshold(waveform, frames)
        else:
            raise ValueError("\nThe type of ground truth is not properly defined!\nExiting...\n")

        for start_time, end_time, _ in boundaries:
            start_idx = int(start_time / FRAME_STRIDE)
            end_idx = int(end_time / FRAME_STRIDE)

            if start_idx < len(labels):
                labels[start_idx] = 1            # marks word start
            if end_idx < len(labels):
                labels[end_idx] = 1             # marks word end

        # Return the features and labels of each samples: --->
        # Features shape: (seq_len, input_dim)
        # Labels shape: (seq_len)
        return tc.tensor(features, dtype = tc.float32), tc.tensor(labels, dtype = tc.float32)

# Custom Collate Function for Variable Length --->
def collate_fn(batch):
    features, labels = zip(*batch)  # list of (seq_len_i, input_dim), (seq_len_i,)

    # Return the padded sequence lengths: --->
    # Padded features shape: (batch, max_seq_len, input_dim)
    # Padded labels shape: (batch, max_seq_len)
    # Return lengths to use packed sequences
    return pad_sequence(features, batch_first = True), pad_sequence(labels, batch_first = True), tc.tensor([len(f) for f in features])
