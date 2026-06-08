import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import yfinance as yf
from typing import Tuple, List

from core_utils.logger import get_logger
from forecasting_pipeline.config import ForecastingConfig

logger = get_logger("StockDataset")

class StockSequenceDataset(Dataset):
    """
    Custom Dataset for processing asset price series into stationary rolling windows.
    Supports multi-horizon forecasting targets and turning point classification.
    """
    def __init__(self, config: ForecastingConfig, target_type: str, split: str = "train"):
        self.config = config
        self.target_type = target_type
        self.split = split
        
        self.feature_tensors = []
        self.target_tensors = []
        self.dates = []
        self.num_features = 0
        
        self._process_pipeline()
        
    def _download_or_load_cache(self, ticker: str) -> pd.DataFrame:
        cache_path = os.path.join(self.config.cache_dir, f"{ticker}.csv")
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        
        logger.info(f"Fetching {ticker} market metrics from Yahoo Finance...")
        df = yf.download(
            ticker,
            start=self.config.start_date,
            end=self.config.end_date,
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        if df is None or df.empty:
            raise RuntimeError(f"Failed to query data partitions for ticker: {ticker}")
            
        # Standardize modern yfinance multi-index columns securely if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[["Open", "High", "Low", "Close"]].dropna()
        df.to_csv(cache_path)
        return df

    def _compute_technical_indicators(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        features = list(self.config.features)
        out = df.copy()
        if self.config.use_ma_feature:
            for window_size in self.config.ma_windows:
                col_name = f"MA_{window_size}"
                out[col_name] = out["Close"].rolling(window_size, min_periods=1).mean()
                features.append(col_name)
        return out, features

    def _calculate_target(self, close_arr: np.ndarray, high_arr: np.ndarray, t: int) -> np.ndarray:
        if self.target_type == "return":
            # Experiment B: Standard d-day return ratio tracking
            return np.array([(close_arr[t + d] - close_arr[t]) / close_arr[t] for d in self.config.horizons])
            
        elif self.target_type == "rolling":
            # Experiment C: Weighted rolling average returns over window length l
            w_roll = self._get_rolling_weights()
            y = []
            for d in self.config.horizons:
                idx = t + d - np.arange(self.config.rolling_l + 1)
                avg = np.dot(w_roll, close_arr[idx])
                y.append((avg - close_arr[t]) / close_arr[t])
            return np.array(y)
            
        elif self.target_type == "turning":
            # Experiment D: Turning point classification thresholding
            for d in self.config.horizons:
                metric = high_arr[t + d] / close_arr[t] if self.config.gamma_mode == "ratio" else (high_arr[t + d] - close_arr[t]) / close_arr[t]
                if metric >= self.config.gamma:
                    return np.array([1.0])
            return np.array([0.0])
            
        raise ValueError(f"Unsupported target format: {self.target_type}")

    def _get_rolling_weights(self) -> np.ndarray:
        if self.config.rolling_weights is not None:
            w = np.asarray(self.config.rolling_weights, dtype=np.float64)
        else:
            w = np.arange(self.config.rolling_l + 1, 0, -1, dtype=np.float64)
        return w / w.sum()

    def _process_pipeline(self):
        X_all, Y_all, date_all = [], [], []
        max_h = max(self.config.horizons)
        
        for ticker in self.config.tickers:
            df = self._download_or_load_cache(ticker)
            df, feat_names = self._compute_technical_indicators(df)
            self.num_features = len(feat_names)
            
            feats = df[feat_names].to_numpy(dtype=np.float64)
            close = df["Close"].to_numpy(dtype=np.float64)
            high = df["High"].to_numpy(dtype=np.float64)
            dates = df.index.to_numpy()
            
            # Bound the loop execution to prevent lookahead indexing errors
            for t in range(self.config.lookback_t - 1, len(df) - max_h):
                window = feats[t - self.config.lookback_t + 1 : t + 1].copy()
                if self.config.normalize == "window_close":
                    window /= (close[t] + 1e-8)
                
                target = self._calculate_target(close, high, t)
                
                X_all.append(window)
                Y_all.append(target)
                date_all.append(dates[t])
                
        # FIXED: Enforce proper float32 primitive array conversions instead of passing structural shape tuples into dtype
        X_arr = np.array(X_all, dtype=np.float32)
        Y_arr = np.array(Y_all, dtype=np.float32)
        dates_arr = pd.to_datetime(date_all)
        
        # Build precise mask boundaries separating training, validation, and testing contexts chronologically
        train_mask = dates_arr <= pd.Timestamp(self.config.train_end)
        val_mask = (dates_arr >= pd.Timestamp(self.config.val_start)) & (dates_arr <= pd.Timestamp(self.config.val_end))
        test_mask = dates_arr >= pd.Timestamp(self.config.test_start)
        
        if self.split == "train":
            mask = train_mask
        elif self.split == "val":
            mask = val_mask
        elif self.split == "test":
            mask = test_mask
        else:
            raise ValueError(f"Unknown split identifier: {self.split}")
            
        self.feature_tensors = torch.from_numpy(X_arr[mask]).float()
        
        if self.target_type == "turning":
            self.target_tensors = torch.from_numpy(Y_arr[mask]).float().squeeze(-1)
        else:
            self.target_tensors = torch.from_numpy(Y_arr[mask]).float()
            
        self.dates = dates_arr[mask]

    def __len__(self):
        return len(self.feature_tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.feature_tensors[idx], self.target_tensors[idx]


def build_loaders(config: ForecastingConfig, target_type: str) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    logger.info("Initializing tensor graph data pipelines...")
    train_ds = StockSequenceDataset(config, target_type, split="train")
    val_ds = StockSequenceDataset(config, target_type, split="val")
    test_ds = StockSequenceDataset(config, target_type, split="test")
    
    pos_weight = 1.0
    if target_type == "turning":
        targets = train_ds.target_tensors.numpy()
        pos = targets.sum()
        neg = len(targets) - pos
        pos_weight = float(np.sqrt(neg / max(pos, 1.0)))
        logger.info(f"Class imbalance calculated. Imbalance scaling factor (sqrt dampened): {pos_weight:.3f}")

    if target_type == "turning" and config.balanced_sampler:
        labels = train_ds.target_tensors.long().numpy()
        class_count = np.bincount(labels, minlength=2).astype(np.float64)
        w_per_class = 1.0 / np.maximum(class_count, 1.0)
        sample_w = torch.as_tensor(w_per_class[labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)

    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)
    
    meta = {
        "num_features": train_ds.num_features,
        "pos_weight": pos_weight
    }
    return train_loader, val_loader, test_loader, meta