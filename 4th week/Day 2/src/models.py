"""
Sequence-to-one forecasting models: Simple RNN, LSTM, and GRU, each
followed by a linear readout head on the last hidden state.
"""
import torch
import torch.nn as nn


class _BaseForecaster(nn.Module):
    def __init__(self, cell, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.rnn = cell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last_step = out[:, -1, :]
        return self.head(last_step)


class RNNForecaster(_BaseForecaster):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__(nn.RNN, input_size, hidden_size, num_layers, dropout)


class LSTMForecaster(_BaseForecaster):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__(nn.LSTM, input_size, hidden_size, num_layers, dropout)


class GRUForecaster(_BaseForecaster):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__(nn.GRU, input_size, hidden_size, num_layers, dropout)


MODEL_REGISTRY = {
    "RNN": RNNForecaster,
    "LSTM": LSTMForecaster,
    "GRU": GRUForecaster,
}
