# Import libraries: --->
import torch as tc
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Import custom modules: --->
# from model import MLP
from model import BiRNN

# Import constants: --->
from const import MODEL_NAME, MODEL

def testing(loader):
    # Instantiation of device, model, loss function: --->
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    model = BiRNN(MODEL).to(device)
    model.load_state_dict(tc.load(MODEL_NAME))
    model.eval()

    # Initialize the epoch metrics lists: --->
    all_preds = []
    all_labels = []
    total_loss = 0

    # L,oad the test subset samples for each batches: --->
    with tc.no_grad():
        for x_batch, y_batch, _ in loader:
            # x_batch shape: (B, T, D)
            # y_batch shape: (B, T)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            preds = model(x_batch).squeeze(-1)           # shape: (B, T)

            # Mask: 1 - Create mask where label != PAD (assumes padding = 0): --->
            mask = (y_batch != 0).float()
            loss = criterion(preds, y_batch)
            loss = (loss * mask).sum() / mask.sum()         # masked mean loss

            preds = (preds > 0.5).float()

            # Save the batch history: --->
            total_loss += loss.item()
            all_preds.extend(preds[mask == 1].cpu().numpy())
            all_labels.extend(y_batch[mask == 1].cpu().numpy())

    # Convert the predictions and labels of the test subset to numpy arrays: --->
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute the epoch metrics: --->
    precision = precision_score(all_labels, all_preds, zero_division = 0)
    recall = recall_score(all_labels, all_preds, zero_division = 0)
    f1 = f1_score(all_labels, all_preds, zero_division = 0)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Test Loss: {total_loss / len(loader):.4f}")
    print(f"  [Test]  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Accuracy: {accuracy:.4f}")
