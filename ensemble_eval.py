# Import libraries: --->
import torch as tc
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Import custom modules: --->
from ensemble_model import EnsembleModel

#! Logit-Level Blending (Linear Blending):
def evaluate(models, dataloader):
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    # Create ensemble
    ensemble_model = EnsembleModel(models).to(device)
    ensemble_model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with tc.no_grad():
        for x_batch, y_batch, _ in dataloader:
            # x_batch shape: (B, T, D)
            # y_batch shape: (B, T)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            outputs = ensemble_model(x_batch)
            probs = tc.sigmoid(outputs)
            preds = (probs > 0.5).float()

           # Mask: 1 - Create mask where label != PAD (assumes padding = 0): --->
            mask = (y_batch != 0).float()

            loss = criterion(probs, y_batch)
            loss = (loss * mask).sum() / mask.sum()         # masked mean loss

            total_loss += loss.item()

            # Collect predictions and labels without padded regions: --->
            for i in range(x_batch.size(0)):
                seq_len = _[i]
                all_preds.extend(preds[i][:seq_len].cpu().numpy())
                all_labels.extend(y_batch[i][:seq_len].cpu().numpy())

    # Convert to numpy arrays: --->
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute evaluation metrics: --->
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"\n--- Evaluation Metrics ---")
    print(f"Test Loss: {total_loss / len(dataloader):.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    return total_loss / len(dataloader), precision, recall, f1, accuracy

#! Logit-Level Attention: --->
def evaluate_attention(models, dataloader):
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    # Assume all models produce logits in (B, T) or (B, T, 1)
    ensemble_model = EnsembleModel(models).to(device)
    ensemble_model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with tc.no_grad():
        for x_batch, y_batch, lengths in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # shapes: (B, T, D), (B, T)

            logits = ensemble_model(x_batch)  # shape: (B, T)
            probs = tc.sigmoid(logits)

            # Binary predictions: --->
            preds = (probs > 0.5).float()

            # Mask: 1 - Create mask where label != PAD (assumes padding = 0): --->
            mask = (y_batch != 0).float()

            loss = criterion(probs, y_batch)
            loss = (loss * mask).sum() / mask.sum()         # masked mean loss

            total_loss += loss.item()

            # Collect predictions and labels without padded regions: --->
            for i in range(x_batch.size(0)):
                seq_len = lengths[i]
                all_preds.extend(preds[i][:seq_len].cpu().numpy())
                all_labels.extend(y_batch[i][:seq_len].cpu().numpy())

    # Convert to numpy arrays: --->
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute evaluation metrics: --->
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)

    print(f"\n--- Evaluation Metrics ---")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    return avg_loss, precision, recall, f1, accuracy
