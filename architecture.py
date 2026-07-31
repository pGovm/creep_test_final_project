"""
This file contains all the architectures that will be used for this project.

5 architectures are being used: RNN, LSTM, GRU, Tranformer-encoder

All hyperparameters are hardcoded and shared between the architectures for easier comparison later on during the report. They are:
For CNN:
Number of Filters = 32
Kernel Size = 3
Number of Convolutional Layers = 2

For RNN, LSTM, GRU:
Hidden Size = 32
Number of Layers = 1

For Transformer-encoder:
Number of Heads = 2
Number of Layers = 1
Embedding Dimension = 16

"""

import torch
import torch.nn as nn

class elongationCNN(nn.Module):
    def __init__(self, input_size, num_filters=32, kernel_size=3, num_conv_layers=2, fc_hidden=None, dropout=0.2):
        super().__init__()

        layers = []
        in_channels = input_size

        for _ in range(num_conv_layers):
            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            in_channels = num_filters

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

        if fc_hidden:
            self.fc = nn.Sequential(nn.Linear(num_filters, fc_hidden), nn.ReLU(), nn.Linear(fc_hidden, 1))
        else:
            self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch_len, seq_len, input_size) --> (batch_len, input_size, seq_len)
        x = self.conv(x) # (batch_len, num_filters, seq_len)
        x = self.pool(x).squeeze(-1) # (batch_len, num_filters, 1) --> (batch_len, num_filters)
        x = self.dropout(x)

        return self.fc(x)

class elongationRNN(nn.Module):
    def __init__(self, input_size, hid_size=32, num_layers=1, fc_hidden=None, dropout=0.2):
        super().__init__()

        self.rnn = nn.RNN(input_size, hid_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        if fc_hidden:
            self.fc = nn.Sequential(nn.Linear(hid_size, fc_hidden), nn.ReLU(), nn.Linear(fc_hidden, 1))
        else:
            self.fc = nn.Linear(hid_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        last_step = output[:, -1, :]
        last_step = self.dropout(last_step)

        return self.fc(last_step)

class elongationLSTM(nn.Module):
    def __init__(self, input_size, hid_size=32, num_layers=1, fc_hidden=None, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hid_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        if fc_hidden:
            self.fc = nn.Sequential(nn.Linear(hid_size, fc_hidden), nn.ReLU(), nn.Linear(fc_hidden, 1))
        else:
            self.fc = nn.Linear(hid_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        last_step  = self.dropout(last_step)

        return self.fc(last_step)

class elongationGRU(nn.Module):
    def __init__(self, input_size, hid_size=32, num_layers=1, fc_hidden=None, dropout=0.2):
        super().__init__()

        self.gru = nn.GRU(input_size, hid_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        if fc_hidden:
            self.fc = nn.Sequential(nn.Linear(hid_size, fc_hidden), nn.ReLU(), nn.Linear(fc_hidden, 1))
        else:
            self.fc = nn.Linear(hid_size, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        last_step = output[:, -1, :]
        last_step = self.dropout(last_step)

        return self.fc(last_step)

class elongationTransformer(nn.Module):
    def __init__(self, input_size, seq_len, embed_dim=16, num_heads=2, num_layers=1, ff_dim=64, dropout=0.2):
        super().__init__()

        self.input_proj = nn.Linear(input_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

        encode_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer=encode_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding
        x = self.transformer(x)
        last_step = x[:, -1, :]

        return self.fc(last_step)

MODEL_CLASSES = {
    "CNN": elongationCNN,
    "RNN": elongationRNN,
    "LSTM": elongationLSTM,
    "GRU": elongationGRU,
    "Transformer": elongationTransformer,
}

def get_model(name, input_size, seq_len=None, **overrides):
    if name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model name '{name}'. Choose from: {list(MODEL_CLASSES.keys())}")

    if name == "Transformer":
        if seq_len is None:
            raise ValueError("Sequence length is required to instantiate the Transformer model.")
        return elongationTransformer(input_size=input_size, seq_len=seq_len, **overrides)

    return MODEL_CLASSES[name](input_size=input_size, **overrides)