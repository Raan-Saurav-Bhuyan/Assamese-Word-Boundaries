# 🎙️ Word Boundary Detection in Assamese Audio Speech Using RNN Ensembling

### 🏛️ Participating Institutes
*   **Indian Institute of Technology Guwahati (IITG)**
    North-Guwahati, Kamrup (R), Assam - 781039
*   **Cotton University (CU)**
    Panbazar, Guwahati, Kamrup (M), Assam - 781001

### 👥 Project Personnel
*   **Supervisor**: Dr. Prithwijit Guha, Associate Professor, Dept. of EEE, IIT-G
*   **Mentor**: Meghali Deka, PhD Scholar, CLST, IIT-G
*   **Authors**: Bipasha Goswami & Raan Saurav Bhuyan, MCA 4th Sem Interns, CU

---

## 📑 Table of Contents
1. [Introduction](#introduction)
2. [Feature Extraction Phase](#feature-extraction-phase)
3. [Model Description](#model-description)
4. [Ensembling Technique](#ensembling-technique)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Local Setup and Running Instructions](#local-setup-and-running-instructions)

---

## 🚀 Introduction
This repository hosts the source code for our major project internship. It focuses on Word Boundary Detection (WBD) in quantized Assamese audio samples. WBD is treated as a highly imbalanced binary classification task where the model predicts the exact frame containing a spoken word boundary.

---

## 🛠️ Feature Extraction Phase

### 1. 🎛️ Sampling & Quantization
Raw Assamese audio files (`.wav`) are first sampled at a target **Sample Rate of 22050 Hz**. The audio is normalized and quantized to 16-bit PCM. To process the sequential nature of audio, the continuous waveform is divided into **overlapping frames** of `10 ms` duration with a stride (hop length) of `5 ms` to capture fine-grained temporal state transitions.

### 2. 📊 Extracted Features
We extract three specific types of features, each feeding differently into the recurrent network pipelines:

*   **Discrete Fourier Transform (DFT)**:
    Extracts the magnitude spectrum of the frames. Using a 256-point FFT configuration, the audio segments are transformed into the frequency domain. We retain the real magnitude portion, capturing the energy distribution across frequency bins.
*   **Discrete Cosine Transform (DCT)**:
    DCT-II is used to obtain a real-valued, compact frequency representation. The audio is mirrored (even-symmetry extension) before applying the FFT, yielding a transformed signal that minimizes boundary artifacts. The output is manually orthonormalized to stabilize training.
*   **Mel-Frequency Cepstral Coefficients (MFCC)**:
    Captures the non-linear human auditory perception. We extract 13 MFCC coefficients per frame (`NUM_MFCC=13`) using `librosa`. By applying a Mel-scale filter bank to the power spectrum and taking the log, we isolate the vocal tract's frequency response from the source excitation.

---

## 🧠 Model Description

To capture the temporal dependencies in the sequential audio frames, we utilize Recurrent Neural Networks (RNNs) specifically designed to process sequential data forward and backward simultaneously.

### 1. 🔁 Bi-Directional LSTM (BiLSTM)
The BiLSTM model utilizes 3 hidden layers with 64 hidden units each. LSTMs contain a cell state and three gates (input, forget, and output) that effectively solve the vanishing gradient problem, allowing the model to remember long-term context (e.g., surrounding silence or phoneme characteristics) when classifying a boundary frame.

### 2. ⚡ Bi-Directional GRU (BiGRU)
The BiGRU model serves as a computationally lighter alternative to the LSTM. Using the same depth and hidden dimension, GRUs merge the forget and input gates into a single "update gate". This usually yields comparable performance on audio tasks while speeding up training and reducing parameter count.

*Both architectures optionally utilize a 1D Convolutional frontend (`Conv1D -> BatchNorm -> ReLU`) to extract local frame correlations before routing the feature maps into the BiRNN block. The output is squashed via a fully connected network to a single sigmoid probability per frame.*

---

## 🤝 Ensembling Technique

To leverage the diverse feature representations learned by different models (e.g., BiLSTM trained on DFT + BiGRU trained on MFCC), we employ **Logit-Level Attention Ensembling**.

### 🧮 Mathematical & Algorithmic Description
Let $M$ be the number of models in the ensemble, and $T$ be the sequence length (number of frames).

1.  **Forward Pass**:
    Each independent model $m$ processes the input frames and produces raw pre-sigmoid logits $L_m \in \mathbb{R}^{T}$.
    The logits are stacked: $\mathbf{L} = [L_1, L_2, ..., L_M] \in \mathbb{R}^{T \times M}$.

2.  **Attention Scoring**:
    An Attention Fully-Connected network evaluates the logits to determine how much "trust" to place in each model's prediction at each frame.

    $$ \mathbf{H} = \text{ReLU}(\mathbf{L} \cdot \mathbf{W}_1 + \mathbf{b}_1) $$
    $$ \mathbf{S} = \mathbf{H} \cdot \mathbf{W}_2 + \mathbf{b}_2 $$

    Where $\mathbf{W}_1 \in \mathbb{R}^{M \times 32}$ and $\mathbf{W}_2 \in \mathbb{R}^{32 \times M}$.

3.  **Softmax Normalization**:
    The scores $\mathbf{S}$ are converted into attention weights $\alpha \in \mathbb{R}^{T \times M}$ across the model dimension:

    $$ \alpha_{t, m} = \frac{\exp(S_{t, m})}{\sum_{k=1}^{M} \exp(S_{t, k})} $$

4.  **Weighted Sum**:
    The final ensembled logit output is the weighted sum of the individual model logits:

    $$ \hat{L}_t = \sum_{m=1}^{M} \alpha_{t, m} \cdot L_{t, m} $$

    This output $\hat{L}_t$ is then passed through a Sigmoid function to obtain the final boundary probability.

*(Note: We also experimentally use Logit-Level Linear Blending which simply averages the logits: $\hat{L}_t = \frac{1}{M} \sum L_{t,m}$)*

---

## 📈 Evaluation Metrics

Given the highly imbalanced nature of the word boundary dataset (few boundary frames, many non-boundary frames), we rely on the following metrics beyond standard Binary Cross Entropy (BCE) Loss:

*   **Precision**: $\frac{TP}{TP + FP}$ - The ratio of correctly predicted boundaries to all predicted boundaries.
*   **Recall**: $\frac{TP}{TP + FN}$ - The model's ability to find all actual ground truth boundaries.
*   **F1-Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$ - The harmonic mean of precision and recall. Our primary metric for model effectiveness.
*   **Accuracy**: Overall frame-level classification correctness (though heavily skewed by silence/non-boundaries).

---

## 💻 Local Setup and Running Instructions

Follow these step-by-step instructions to replicate the environment and run the code on your local system.

### 1. ⚙️ Environment Setup
Ensure you have Python 3.8+ installed.

**Option A: Using Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate wbd_env
```
**Option B: Using Pip**
```bash
pip install -r requirements.txt
```
*(Note: If using CUDA, ensure you install the PyTorch build that matches your NVCC/CUDA driver version).*

### 2. Dataset Preparation
Place your Assamese dataset inside the `ASS/` directory so it looks like this:
*   `ASS/audios/` (Contains `.wav` files)
*   `ASS/Textgrid/` (Contains `.TextGrid` Praat files)

Run the alignment script to convert TextGrid tiers to model-readable `.align` files:
```bash
python extract_word_align.py
```
*(This generates `.align` labels and saves them in `ASS/textgrid/`)*

### 3. Feature Pre-Computation
To compute and cache the MFCCs (saving training time):
```bash
python compute_mfcc.py
```

### 4. Training Individual Models
Open `const.py` and set your desired configurations (e.g., `MODEL = "BiLSTM"`, `FEATURES = "DFT"`, `THRESHOLD = "N"`).

Start the training loop:
```bash
python main.py
```
*The best model weights will be saved to the `checkpoints/` directory based on validation loss.*

### 5. Ensemble Evaluation
Once you have trained multiple models (e.g., a BiLSTM and a BiGRU), edit `ensemble.py` and `ensemble_eval.py` to point to your saved checkpoint files.

Run the ensembling evaluation:
```bash
python ensemble.py
```

### 6. Visualizing Predictions
To view the waveforms, ground truths, and model predictions overlaid on matplotlib graphs:

For individual models:
```bash
python plot.py
```

For the attention ensemble:
```bash
python ensemble_plot.py
```
