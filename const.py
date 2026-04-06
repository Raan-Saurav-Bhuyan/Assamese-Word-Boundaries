# Model parameters: --->
INPUT_DIM = 256
HIDDEN_SIZE = 64
HIDDEN_LAYERS = 3

# Audio processing constants: --->
NUM_FFT = INPUT_DIM * 2
NUM_MFCC = 13
SAMPLE_RATE = 22050
FRAME_SIZE = 0.01
FRAME_STRIDE = 0.005

# Hyperparameters: --->
EPOCHS = 800
BATCH_SIZE = 16
LR = 1e-3

# Thresholding parameters: --->
MIN_WORD_DURATION = 0.001
AMPLITUDE_THRESHOLD = 52e-3                 # GRID = 25e-7, ASS = 52e-3
ENERGY_THRESHOLD = 3e-1                       # GRID = 20e-1, ASS = 3e-1

# Directory constants: --->
DATASET_ROOT = "ASS"
DATASET_SAMPLES = "audios"
DATASET_LABELS = "textgrid"
MFCC_ROOT = "MFCC"
MODEL_ROOT = "checkpoints"
THRESHOLD = "N"                 # A for amplitude, E for energy, N for none
FEATURES = "DFT"                 # DFT for DFT, MFCC for MFCC, DCT for DCT
MODEL = "BiLSTM"                 # BiGRU or BILSTM
CONV = False
BIDIRECTIONAL = True
if CONV:
    MODEL_NAME = f"{MODEL_ROOT}/wbd_{DATASET_ROOT.lower()}_{THRESHOLD.lower()}t_{FEATURES.lower()}_conv_{MODEL.lower()}_1.pth"
else:
    MODEL_NAME = f"{MODEL_ROOT}/wbd_{DATASET_ROOT.lower()}_{THRESHOLD.lower()}t_{FEATURES.lower()}_{MODEL.lower()}_1.pth"
