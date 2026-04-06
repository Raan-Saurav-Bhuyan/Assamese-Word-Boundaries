# Import libraries: --->
import os
import numpy as np
import torch as tc
import torch.nn as nn
from torch import nn, optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Import custom modules: --->
# from model import MLP
from model import BiRNN

# Import constants: --->
from const import EPOCHS, MODEL_ROOT, MODEL_NAME, LR, MODEL

def training(loader):
    # Instantiation of device, model, loss function and optimizer: --->
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    model = BiRNN(MODEL).to(device)
    optimizer = optim.Adam(model.parameters(), lr = LR)

    total_loss = []

    os.makedirs(MODEL_ROOT, exist_ok=True)

    # Train the model over the epochs: --->
    for epoch in range(EPOCHS):
        model.train()

        # Initialize the epoch metrics lists: --->
        all_preds = []
        all_labels = []
        epoch_loss = 0
        minimum  = 0

        # L,oad the train subset sample batches for each epoch: --->
        for x_batch, y_batch, _ in loader:
            # x_batch shape: (B, T, D)
            # y_batch shape: (B, T)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            preds = model(x_batch).squeeze(-1)               #shape: (B, T)

            # Mask: 1 - Create mask where label != PAD (assumes padding = 0): --->
            mask = (y_batch != 0).float()
            loss = criterion(preds, y_batch)
            loss = (loss * mask).sum() / mask.sum()          # masked mean loss

            # Back-propagation: --->
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (preds > 0.5).float()

            # Save the batch history: --->
            epoch_loss += loss.item()
            all_preds.extend(preds[mask == 1].cpu().numpy())
            all_labels.extend(y_batch[mask == 1].cpu().numpy())

        # Convert the predictions and labels of the train subset to numpy arrays: --->
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute the epoch metrics: --->
        precision = precision_score(all_labels, all_preds, zero_division = 0)
        recall = recall_score(all_labels, all_preds, zero_division = 0)
        f1 = f1_score(all_labels, all_preds, zero_division = 0)
        accuracy = accuracy_score(all_labels, all_preds)

        if epoch != 0:
            minimum = min(total_loss)

        total_loss.append(epoch_loss / len(loader))

        print(f"Epoch {epoch+1} | Train Loss: {total_loss[-1]:.4f}")
        print(f"  [Train]  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Accuracy: {accuracy:.4f}")

        # Save the model every 10 epochs: --->
        if epoch != 0 and total_loss[-1] < minimum:
            print(f"Loss improved from {minimum:.4f} to {total_loss[-1]:.4f}. Saving the model as: {MODEL_NAME}.")
            tc.save(model.state_dict(), MODEL_NAME)
