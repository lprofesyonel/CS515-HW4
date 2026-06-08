import torch
import torch.nn as nn
import torch.nn.functional as F
from comm_system.config import CommConfig
from comm_system.channel import AWGNChannel

class LearnedPositionalEncoding(nn.Module):
    """
    Trainable absolute positional embeddings added to the input sequences.
    """
    def __init__(self, sequence_length: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class TransformerBlock(nn.Module):
    """
    Standard Post-LN Transformer Block matching Equations (1)-(3) from the homework.
    Applies LayerNormalization right after the residual connections.
    """
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Equation (2): Multi-Head Attention followed by residual connection and LayerNorm
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.layer_norm1(x + self.dropout_layer(attn_out))
        
        # Equation (3): Feed-Forward Network followed by residual connection and LayerNorm
        x = self.layer_norm2(x + self.dropout_layer(self.ffn(x)))
        return x


class TransformerStack(nn.Module):
    """
    A stack of identical Transformer layers preceded by positional encodings.
    """
    def __init__(self, n_layers: int, d_model: int, n_heads: int, ffn_dim: int, dropout: float, sequence_length: int):
        super().__init__()
        self.pe = LearnedPositionalEncoding(sequence_length, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Equation (1): Add positional encoding to the initial preprocessing representation
        hidden_state = self.pe(hidden_state)
        for block in self.blocks:
            hidden_state = block(hidden_state)
        return hidden_state


class TXEncoder(nn.Module):
    """
    Transmitter network that maps current message symbols, previous transmissions,
    received feedback, and round information to a single channel symbol per sequence element.
    """
    def __init__(self, config: CommConfig):
        super().__init__()
        self.config = config
        
        # Input features per token: one-hot message + tx history + feedback history + round indicator
        input_dim = config.alphabet + 3 * config.t_rounds
        
        # Hint 3: MLP before and after the transformer block
        self.pre_mlp = nn.Sequential(
            nn.Linear(input_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        self.transformer = TransformerStack(
            config.n_layers, config.d_model, config.n_heads, config.ffn_dim, config.dropout, config.n_symbols
        )
        self.post_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1) # Outputs a 1D coded symbol per token
        )

    def forward(self, source_onehot: torch.Tensor, tx_history: torch.Tensor, fb_history: torch.Tensor, round_idx: int) -> torch.Tensor:
        batch_size, seq_len, _ = source_onehot.shape
        
        # Create a one-hot indicator vector for the current communication round
        round_indicator = torch.zeros(batch_size, seq_len, self.config.t_rounds, device=source_onehot.device)
        round_indicator[:, :, round_idx] = 1.0
        
        # Concatenate features to form the raw sequence token matrix Z^(t)
        z_t = torch.cat([source_onehot, tx_history, fb_history, round_indicator], dim=-1)
        
        # Forward pass through the transmitter components
        h = self.pre_mlp(z_t)
        h = self.transformer(h)
        x_t = self.post_mlp(h).squeeze(dim=-1)
        
        # Apply average power constraint normalization E[||x||^2] <= 1
        return AWGNChannel.power_normalize(x_t)


class RXDecoder(nn.Module):
    """
    Receiver network that processes the accumulated sequence trajectories
    over all communication rounds to output reconstruction logits.
    """
    def __init__(self, config: CommConfig):
        super().__init__()
        self.config = config
        
        # Input feature size is t_rounds (trajectories of length T)
        self.pre_mlp = nn.Sequential(
            nn.Linear(config.t_rounds, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        self.transformer = TransformerStack(
            config.n_layers, config.d_model, config.n_heads, config.ffn_dim, config.dropout, config.n_symbols
        )
        self.post_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.alphabet) # Logits over the alphabet size
        )

    def forward(self, channel_outputs: torch.Tensor) -> torch.Tensor:
        h = self.pre_mlp(channel_outputs)
        h = self.transformer(h)
        return self.post_mlp(h)


class FeedbackTransceiver(nn.Module):
    """
    Joint Autoencoder that orchestrates the multi-round communication loop.
    Tracks channel history and computes final predictions.
    """
    def __init__(self, config: CommConfig):
        super().__init__()
        self.config = config
        self.transmitter = TXEncoder(config)
        self.receiver = RXDecoder(config)

    def forward(self, source_symbols: torch.Tensor, override_sigma: float = None):
        sigma = override_sigma if override_sigma is not None else self.config.sigma
        batch_size, seq_len = source_symbols.shape
        device = source_symbols.device
        
        # Convert message symbols to one-hot encoding
        source_onehot = F.one_hot(source_symbols, self.config.alphabet).float()
        
        # Buffers to track transmission and feedback histories across rounds
        tx_history = torch.zeros(batch_size, seq_len, self.config.t_rounds, device=device)
        fb_history = torch.zeros(batch_size, seq_len, self.config.t_rounds, device=device)
        channel_outputs = torch.zeros(batch_size, seq_len, self.config.t_rounds, device=device)

        # Multi-round communication loop
        for t in range(self.config.t_rounds):
            x_t = self.transmitter(source_onehot, tx_history, fb_history, t)
            y_t = AWGNChannel.add_noise(x_t, sigma)
            
            # Hint 1: Simple noiseless relay mechanism (feedback = received noisy symbol)
            f_t = y_t  
            
            # Update history logs using clone to prevent in-place autograd errors
            tx_history = tx_history.clone()
            tx_history[:, :, t] = x_t
            
            fb_history = fb_history.clone()
            fb_history[:, :, t] = f_t
            
            channel_outputs[:, :, t] = y_t

        # Hint 2: Execute the receiver decoder only at the end of all rounds
        logits = self.receiver(channel_outputs)
        return logits, channel_outputs