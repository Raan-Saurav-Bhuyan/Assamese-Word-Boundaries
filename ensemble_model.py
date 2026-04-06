# Import libraries: --->
import torch as tc
import torch.nn as nn

# Ensemble wrapper: --->
class EnsembleModel(tc.nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

        self.attention_fc = nn.Sequential(
            nn.Linear(self.num_models, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_models),
            nn.Softmax(dim=1)
        )

    #! Only for Logit-Level Blending (Linear Blending): --->
    # def forward(self, x):
    #     predictions = []

    #     for model in self.models:
    #         model.eval()

    #         with tc.no_grad():
    #             preds = model(x)
    #             predictions.append(preds.unsqueeze(0))            # Shape: (1, B, T)

    #     predictions = tc.cat(predictions, dim = 0)                   # Shape: (N_models, B, T)
    #     avg_preds = predictions.mean(dim = 0)                     # Shape: (B, T)

    #     return avg_preds

    #! Only for Logit-Level Attention: --->
    def forward(self, x):
        # Forward through all models: --->
        logits = tc.stack([model(x) for model in self.models], dim = 1)             # Shape: (B, M, T)

        # Transpose for attention over models: --->
        logits = logits.permute(0, 2, 1)                                                            # Shape: (B, T, M)

        # Apply attention to the weights: --->
        attn_weights = self.attention_fc(logits)                                               # Shape: (B, T, 1)
        attn_weights = tc.softmax(attn_weights, dim = -1)                               # optional

        # Weighted sum: --->
        attended = (attn_weights * logits).sum(dim = -1)                              # Shape: (B, T)

        return attended
