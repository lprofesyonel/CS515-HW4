import os
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any

from core_utils.trainer import BaseTrainer
from comm_system.config import CommConfig
from core_utils.logger import get_logger

logger = get_logger("EndToEndCommTrainer")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """
    Creates a learning rate scheduler with a linear warmup followed by a cosine decay.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EndToEndCommTrainer(BaseTrainer):
    """
    Trainer for the multi-round feedback communication system. 
    Optimizes transmitter and receiver jointly using Block Error Rate (BLER) tracking.
    """
    def __init__(self, model, optimizer, config: CommConfig):
        super().__init__(model, optimizer, config.torch_device, config)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.config.warmup, self.config.steps)
        self.global_step = 0

    def _generate_random_messages(self, batch_size: int) -> torch.Tensor:
        """
        Generates random message symbols uniformly distributed over the alphabet size.
        Output shape: (batch_size, n_symbols)
        """
        return torch.randint(0, self.config.alphabet, (batch_size, self.config.n_symbols), device=self.device)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        
        # Train for 'eval_every' steps before running a validation check
        for _ in range(self.config.eval_every):
            self.global_step += 1
            source_symbols = self._generate_random_messages(self.config.batch_size)
            
            # Forward pass through the joint transceiver system
            logits, _ = self.model(source_symbols)
            
            # Reshape tensors to compute standard cross-entropy loss over the sequences
            loss = F.cross_entropy(logits.reshape(-1, self.config.alphabet), source_symbols.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
        return {"loss": total_loss / self.config.eval_every}

    @torch.no_grad()
    def _evaluate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        symbol_errors = 0
        block_errors = 0
        total_symbols = 0
        total_blocks = 0
        
        for _ in range(self.config.eval_batches):
            source_symbols = self._generate_random_messages(self.config.batch_size)
            logits, _ = self.model(source_symbols)
            
            loss = F.cross_entropy(logits.reshape(-1, self.config.alphabet), source_symbols.reshape(-1))
            total_loss += loss.item()
            
            # Performance metrics tracking (SER and BLER)
            predicted_symbols = logits.argmax(dim=-1)
            mismatches = (predicted_symbols != source_symbols)
            
            symbol_errors += mismatches.sum().item()
            block_errors += mismatches.any(dim=-1).sum().item() # If any symbol in the block is wrong, block is wrong
            
            total_symbols += source_symbols.numel()
            total_blocks += source_symbols.size(0)
            
        ser = symbol_errors / max(total_symbols, 1)
        bler = block_errors / max(total_blocks, 1)
        
        return {
            "loss": total_loss / self.config.eval_batches,
            "ser": ser,
            "bler": bler
        }

    def _is_better(self, current: float, best: float) -> bool:
        # Since we track BLER, lower values are better
        return current < best

    def train(self, target_metric_name: str, maximize_metric: bool = False, patience: int = 10, save_path: str = "checkpoints/best_model.pt"):
        # Convert total training steps into epoch counts based on the evaluation intervals
        self.config.epochs = self.config.steps // self.config.eval_every
        super().train(target_metric_name, maximize_metric, patience=patience, save_path=save_path)
        
    @torch.no_grad()
    def evaluate_snr_sweep(self) -> Dict[float, Dict[str, float]]:
        """
        Evaluates system performance across a range of SNR configurations.
        """
        self.model.eval()
        results = {}
        snrs = torch.linspace(self.config.eval_snr_min, self.config.eval_snr_max, self.config.eval_snr_steps)
        
        logger.info("Starting performance evaluation sweep across multiple SNR values...")
        for snr in snrs:
            snr_db = snr.item()
            
            # Convert SNR from dB to linear scale to extract the standard deviation (sigma)
            snr_linear = 10.0 ** (snr_db / 10.0)
            sigma = 1.0 / (snr_linear ** 0.5)
            
            symbol_errors = 0
            block_errors = 0
            total_symbols = 0
            total_blocks = 0
            
            for _ in range(self.config.eval_batches):
                source_symbols = self._generate_random_messages(self.config.batch_size)
                logits, _ = self.model(source_symbols, override_sigma=sigma)
                
                predicted_symbols = logits.argmax(dim=-1)
                mismatches = (predicted_symbols != source_symbols)
                
                symbol_errors += mismatches.sum().item()
                block_errors += mismatches.any(dim=-1).sum().item()
                
                total_symbols += source_symbols.numel()
                total_blocks += source_symbols.size(0)
                
            ser = symbol_errors / max(total_symbols, 1)
            bler = block_errors / max(total_blocks, 1)
            results[snr_db] = {"ser": ser, "bler": bler}
            logger.info(f"SNR: {snr_db:5.1f} dB | SER: {ser:.5f} | BLER: {bler:.5f}")
            
        return results