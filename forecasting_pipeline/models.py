import torch
import torch.nn as nn
from forecasting_pipeline.config import ForecastingConfig

class ForecastingLSTM(nn.Module):
    """
    Unidirectional LSTM network mapped to multi-horizon return regression.
    
    Input shape:  (batch_size, lookback_t, num_features)
    Output shape: (batch_size, d_out) -> Predicted returns for each horizon d
    """
    def __init__(self, num_features: int, config: ForecastingConfig):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=num_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )
        self.regressor = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.d_out)
        )

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(feature_tensor)
        # Extract temporal boundary representation from the terminal step (T)
        terminal_state = out[:, -1, :]
        return self.regressor(terminal_state)


class ForecastingGRU(nn.Module):
    """
    Gated Recurrent Unit network mapped to multi-horizon return regression.
    
    Input shape:  (batch_size, lookback_t, num_features)
    Output shape: (batch_size, d_out) -> Predicted returns for each horizon d
    """
    def __init__(self, num_features: int, config: ForecastingConfig):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=num_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )
        self.regressor = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.d_out)
        )

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(feature_tensor)
        # Extract temporal boundary representation from the terminal step (T)
        terminal_state = out[:, -1, :]
        return self.regressor(terminal_state)


class TurningPointBiRNN(nn.Module):
    """
    Bidirectional Recurrent topology (LSTM/GRU) for binary turning point classification.
    Concatenates the terminal forward hidden state and the initial backward hidden state.
    
    Input shape:  (batch_size, lookback_t, num_features)
    Output shape: (batch_size,) -> Raw unnormalized logits (No Sigmoid applied here)
    """
    def __init__(self, num_features: int, config: ForecastingConfig, cell_type: str = "lstm"):
        super().__init__()
        rnn_type = nn.LSTM if cell_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_type(
            input_size=num_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1)
        )

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(feature_tensor)
        hidden_dim = out.size(-1) // 2
        
        # Isolate final state of forward pass (at index -1) and final state of backward pass (at index 0)
        fwd_final = out[:, -1, :hidden_dim]
        bwd_final = out[:, 0, hidden_dim:]
        
        # Fuse contexts into a unified sequence embedding
        context_vector = torch.cat([fwd_final, bwd_final], dim=-1)
        logits = self.classifier(context_vector)
        
        # Return squeeze representation safe for binary loss optimization graphs
        return logits.squeeze(dim=-1)