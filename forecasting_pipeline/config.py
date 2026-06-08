import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class ForecastingConfig:
    # Execution parameters
    seed: int = 42
    device: str = "cuda"
    mode: str = "both"          # choices: 'train', 'test', 'both'
    experiment: str = "all"      # choices: 'b', 'c', 'd', 'all'
    arch: str = "both"          # choices: 'lstm', 'gru', 'both'

    # Data pipeline settings (Strictly mapped to SU CS515 HW4 requirements)
    tickers: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    train_end: str = "2024-07-31"
    val_start: str = "2024-08-01"
    val_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    features: List[str] = field(default_factory=lambda: ["Open", "High", "Low", "Close"])
    use_ma_feature: bool = True
    ma_windows: List[int] = field(default_factory=lambda: [5, 10])
    normalize: str = "window_close"

    # Forecasting configurations
    lookback_t: int = 20
    horizons: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    # Target smoothing features (Experiment C)
    rolling_l: int = 3
    rolling_weights: Optional[List[float]] = None

    # Structural classification settings (Experiment D)
    gamma: float = 1.1
    gamma_mode: str = "ratio"    # choices: 'ratio', 'return'
    loss_type: str = "focal"     # choices: 'bce', 'focal'
    pos_weight_scale: float = 0.3
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    select_metric: str = "ap"    # choices: 'loss', 'f1', 'ap'
    tune_threshold: bool = True
    balanced_sampler: bool = True

    # Network topology
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Optimization hyper-parameters
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    grad_clip: float = 1.0
    patience: int = 12

    def __post_init__(self):
        # Resolve absolute execution paths based on local context
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.base_dir, "checkpoints") # Aligned with project structure
        self.plots_dir = os.path.join(self.base_dir, "plots")
        self.cache_dir = os.path.join(self.base_dir, "data_cache")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Automatically derive deterministic uniform weights for Experiment C if not supplied
        if self.rolling_weights is None:
            weight_count = self.rolling_l + 1  # For l=3, j ranges from 0 to 3 (4 elements)
            self.rolling_weights = [1.0 / weight_count] * weight_count

    @property
    def d_out(self) -> int:
        return len(self.horizons)

    @property
    def torch_device(self) -> torch.device:
        if self.device.lower() == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")