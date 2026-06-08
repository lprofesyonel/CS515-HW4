import torch
import torch.nn as nn

class AWGNChannel:
    """
    Simulates an Additive White Gaussian Noise (AWGN) channel with a strict
    average power constraint on the transmitted signals.
    """
    @staticmethod
    def power_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalizes the transmitted signal x to ensure it satisfies the power constraint:
        E[||x||^2] <= 1 per communication round.
        
        Expected input shape: (batch_size, 4) where 4 is the sequence length (number of symbols).
        """
        # Calculate the average energy (L2 norm squared) per block across the entire batch
        # torch.sum(..., dim=-1) gives ||x||^2 for each sample, then torch.mean averages over the batch.
        mean_energy = torch.mean(torch.sum(x ** 2, dim=-1))
        
        # Scaling ensures E[||x_normalized||^2] equals exactly 1.0
        return x / torch.sqrt(mean_energy + eps)

    @staticmethod
    def add_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Adds zero-mean Gaussian noise with standard deviation 'sigma' to the signal.
        For sigma^2 = 0.25, pass sigma = 0.5.
        """
        noise = torch.randn_like(x) * sigma
        return x + noise