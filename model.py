import torch.nn as nn

# Import custom constants: --->
from const import INPUT_DIM, HIDDEN_SIZE, HIDDEN_LAYERS, BIDIRECTIONAL, MODEL, CONV

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class BiRNN(nn.Module):
    def __init__(self, model = MODEL):
        super(BiRNN, self).__init__()
        self.model = model

        if CONV:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels = INPUT_DIM, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(in_channels =  64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                # nn.MaxPool1d(kernel_size = 2)
            )

        if self.model == 'BiGRU':
            self.rnn = nn.GRU(
                input_size = 128 if CONV else INPUT_DIM,
                hidden_size = HIDDEN_SIZE,
                num_layers = HIDDEN_LAYERS,
                batch_first = True,
                bidirectional = BIDIRECTIONAL
            )
        elif self.model == 'BiLSTM':
            self.rnn = nn.LSTM(
                input_size = 128 if CONV else INPUT_DIM,
                hidden_size = HIDDEN_SIZE,
                num_layers = HIDDEN_LAYERS,
                batch_first = True,
                bidirectional = BIDIRECTIONAL
            )
        else:
            raise ValueError(f"MODEL must be either 'BiGRU' or 'BILSTM'!\nCurrent value of MODEL = {MODEL}.")

        # Initialization of the sequential fully connected layers: --->
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2 if BIDIRECTIONAL else HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()                # For binary classification (boundary or not)
        )

    def forward(self, x):
        # x: (batch_size, sequence_len, input_dim): --->
        # assert len(x.shape) == 3, f"Expected input shape (B, T, D), got {x.shape}"

        if CONV:
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)

        # Different outputs of the models: --->
        # (1) BiLSTM output: (batch, seq_len, hidden * 2)
        # (2) BiGRU output: (batch, seq_len, hidden * 2)
        # (3) FC outout: (batch, seq_len, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)

        return x.squeeze(-1)             # (batch, seq_len)
