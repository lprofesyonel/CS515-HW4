import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
from sklearn.metrics import average_precision_score, f1_score

from core_utils.trainer import BaseTrainer
from forecasting_pipeline.config import ForecastingConfig

def focal_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    """
    Binary Focal Loss formulation for stabilizing heavily imbalanced target distributions.
    Utilizes stable log-sum-exp tracking via native BCE framework.
    """
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    modulating_factor = (1.0 - p_t) ** gamma
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * modulating_factor * ce_loss).mean()

class ForecastingTrainer(BaseTrainer):
    """
    Optimized optimization wrapper for sequential quantitative frameworks.
    Encapsulates Multi-Horizon MSE loss boundaries and directional classification pipelines.
    """
    def __init__(self, model, optimizer, train_loader, val_loader, config: ForecastingConfig, task_type: str, pos_weight: float = 1.0):
        super().__init__(model, optimizer, config.torch_device, config)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_type = task_type
        
        # Initialize default operational threshold for classification tasks
        self.best_threshold = 0.5
        
        if self.task_type == "turning":
            scaled_pw = pos_weight * config.pos_weight_scale
            self.pos_weight_tensor = torch.tensor([scaled_pw], device=self.device)
        else:
            self.criterion = nn.MSELoss()

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.task_type == "turning":
            if self.config.loss_type == "focal":
                return focal_loss_with_logits(predictions, targets, self.config.focal_alpha, self.config.focal_gamma)
            return F.binary_cross_entropy_with_logits(predictions, targets, pos_weight=self.pos_weight_tensor)
        return self.criterion(predictions, targets)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss, num_samples = 0.0, 0
        directional_hits = 0
        
        for features, targets in self.train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self._compute_loss(predictions, targets)
            loss.backward()
            
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            batch_sz = features.size(0)
            total_loss += loss.item() * batch_sz
            num_samples += batch_sz
            
            with torch.no_grad():
                if self.task_type == "turning":
                    preds = (torch.sigmoid(predictions) >= self.best_threshold).float()
                    directional_hits += (preds == targets).sum().item()
                else:
                    directional_hits += ((predictions > 0) == (targets > 0)).sum().item()
                
        denominator = num_samples if self.task_type == "turning" else num_samples * self.config.d_out
        return {
            "loss": total_loss / max(num_samples, 1),
            "accuracy": directional_hits / max(denominator, 1)
        }

    @torch.no_grad()
    def _evaluate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss, num_samples = 0.0, 0
        directional_hits = 0
        all_preds, all_targets = [], []
        
        for features, targets in self.val_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            predictions = self.model(features)
            loss = self._compute_loss(predictions, targets)
            
            batch_sz = features.size(0)
            total_loss += loss.item() * batch_sz
            num_samples += batch_sz
            
            if self.task_type == "turning":
                all_preds.append(torch.sigmoid(predictions).cpu())
                all_targets.append(targets.cpu())
            else:
                directional_hits += ((predictions > 0) == (targets > 0)).sum().item()
            
        metrics = {
            "loss": total_loss / max(num_samples, 1)
        }
        
        if self.task_type == "turning" and len(all_targets) > 0:
            y_true = torch.cat(all_targets).numpy()
            y_prob = torch.cat(all_preds).numpy()
            
            # Dynamic threshold optimization layer if enabled during validation epochs
            if self.config.tune_threshold and epoch > 0:
                best_f1 = -1.0
                # Scan precision horizons to maximize F1 distribution boundaries safely
                for candidate_thresh in np.linspace(0.1, 0.9, 81):
                    score = f1_score(y_true, (y_prob >= candidate_thresh).astype(int), zero_division=0)
                    if score > best_f1:
                        best_f1 = score
                        self.best_threshold = candidate_thresh
            
            # Apply optimized boundary threshold for final classification maps
            y_pred = (y_prob >= self.best_threshold).astype(int)
            directional_hits = (y_pred == y_true).sum()
            
            metrics["accuracy"] = directional_hits / max(num_samples, 1)
            if 0 < y_true.sum() < len(y_true):
                metrics["ap"] = average_precision_score(y_true, y_prob)
                metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
            else:
                metrics["ap"] = 0.0
                metrics["f1"] = 0.0
        else:
            denominator = num_samples * self.config.d_out
            metrics["accuracy"] = directional_hits / max(denominator, 1)
                
        return metrics

    def _is_better(self, current: float, best: float) -> bool:
        if self.task_type == "turning" and self.config.select_metric in ("ap", "f1"):
            return current > best
        return current < best