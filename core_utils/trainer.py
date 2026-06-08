import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from core_utils.logger import get_logger

logger = get_logger("BaseTrainer")


class BaseTrainer(ABC):
    """
    Abstract baseline orchestration controller managing model optimization, 
    checkpoint persistence, and performance evaluation pipelines.
    """
    def __init__(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer], device: torch.device, config: Any):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.history = []
        self.best_metric = None
        self.epochs_without_improvement = 0

    @abstractmethod
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Executes a single forward/backward optimization pass across the training partition.
        """
        pass

    @abstractmethod
    def _evaluate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Evaluates model generalization capabilities across the validation boundary.
        """
        pass

    def _save_checkpoint(self, path: str, extra_state: Optional[Dict[str, Any]] = None):
        """
        Serializes model parameters and optimizer states to persistent disk storage.
        """
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
        }
        if extra_state:
            state.update(extra_state)
            
        torch.save(state, path)
        logger.info(f"Synchronized checkpoint written securely to: {path}")

    def _load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Restores computation state tensors from physical storage binaries.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Target checkpoint binary not resolved at: {path}")
            
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        if "optimizer_state" in state and state["optimizer_state"] is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer_state"])
            
        logger.info(f"Successfully operationalized state parameters from: {path}")
        return state

    def train(self, target_metric_name: str, maximize_metric: bool = False, patience: int = 10, save_path: str = "checkpoints/best_model.pt"):
        """
        Executes deterministic multi-epoch iterative training loops with automated 
        early-stopping monitors and state restorations.
        """
        self.best_metric = -float("inf") if maximize_metric else float("inf")
        checkpoint_saved = False

        # Safely resolve standard epoch/step configuration limits to preserve multi-pipeline compatibility
        total_loops = getattr(self.config, "epochs", None)
        if total_loops is None:
            total_loops = getattr(self.config, "steps", 1)

        for epoch in range(1, total_loops + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._evaluate_epoch(epoch)

            # Consolidate telemetry indicators across partition boundaries
            combined_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
            combined_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            combined_metrics["epoch"] = epoch
            self.history.append(combined_metrics)

            if target_metric_name not in val_metrics:
                raise KeyError(f"Target optimization metric '{target_metric_name}' absent from evaluation telemetry.")

            val_target = val_metrics[target_metric_name]
            
            # Unify convergence evaluation logic directly from structural flags
            if maximize_metric:
                improved = val_target > self.best_metric
            else:
                improved = val_target < self.best_metric

            if improved:
                self.best_metric = val_target
                self.epochs_without_improvement = 0
                checkpoint_saved = True
                logger.info(f"Epoch {epoch}: {target_metric_name} advanced convergence boundary to {self.best_metric:.5f}")
                self._save_checkpoint(save_path, extra_state={"best_metric": self.best_metric, "history": self.history})
            else:
                self.epochs_without_improvement += 1
                logger.info(f"Epoch {epoch}: {target_metric_name} stagnated (Current: {val_target:.5f} | Historical Best: {self.best_metric:.5f})")

            if patience > 0 and self.epochs_without_improvement >= patience:
                logger.warning(f"Early-stopping threshold breached. Terminating loop at epoch {epoch}")
                break

        # Safe and robust restoration from persistent storage to avoid tracking or reference bugs
        if checkpoint_saved and os.path.exists(save_path):
            self._load_checkpoint(save_path)
            logger.info("Restored non-stagnated peak performance parameters to global module context.")
        else:
            logger.warning("No optimized checkpoint state synchronized during training execution. Retaining current states.")