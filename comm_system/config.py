import os
import torch
from dataclasses import dataclass

@dataclass
class CommConfig:
    # Execution settings
    seed: int = 42
    device: str = "cuda"
    mode: str = "both"           # choices: 'train', 'test', 'both'
    
    # Core channel & interaction parameters
    alphabet: int = 8            # Cardinality of the message alphabet (S_0 = {1, ..., 8})
    n_symbols: int = 4           # Message sequence length (Strictly 4 symbols per homework)
    t_rounds: int = 4            # Total communication/interaction rounds (T = 4)
    noise_var: float = 0.25      # Channel noise variance (sigma^2 = 0.25 per homework)
    
    # Evaluation SNR sweep boundaries (dB)
    eval_snr_min: float = -2.0   
    eval_snr_max: float = 8.0    
    eval_snr_steps: int = 11     
    
    # Transformer backbone architecture
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.0

    # Optimization & schedule hyperparameters
    batch_size: int = 1024
    steps: int = 50000
    warmup: int = 5000
    eval_every: int = 1000
    eval_batches: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    def __post_init__(self):
        # Setup run directories automatically
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.base_dir, "models_ckpt")
        self.plots_dir = os.path.join(self.base_dir, "plots")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    @property
    def sigma(self) -> float:
        """Channel noise standard deviation (sqrt of variance)."""
        return self.noise_var ** 0.5

    @property
    def torch_device(self) -> torch.device:
        if self.device.lower() == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")