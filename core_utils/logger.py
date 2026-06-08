import logging
import sys
import time
from typing import Dict, Any, Optional

def get_logger(name: str) -> logging.Logger:
    """
    Instantiates a strictly isolated console logger with dedicated stream handlers.
    Disables propagation to prevent duplicate stream mirroring across execution contexts.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class ProgressTracker:
    """
    Lightweight, deterministic CLI progress tracker optimizing terminal refresh cycles.
    Designed for synchronous training step telemetry without external overhead.
    """
    def __init__(self, total_steps: int, prefix: str = "", length: int = 40):
        self.total_steps = max(1, total_steps)  # Guard against division by zero
        self.prefix = prefix
        self.length = length
        self.start_time = time.time()
        
    def step(self, current_step: int, metrics: Optional[Dict[str, Any]] = None):
        """
        Updates the inline ASCII progress track with formatted metrics telemetry.
        """
        if current_step > self.total_steps:
            current_step = self.total_steps
            
        progress = current_step / float(self.total_steps)
        filled_length = int(self.length * progress)
        bar = '=' * filled_length + '-' * (self.length - filled_length)
        
        elapsed = time.time() - self.start_time
        
        # Build optimized metrics string representation
        metrics_payload = metrics or {}
        metrics_str = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
            for k, v in metrics_payload.items()
        )
        
        # Format track alignment with tailing carriage flush
        suffix = f" | {metrics_str}" if metrics_str else ""
        
        # \033[K clears the line from the cursor position to the end to prevent text ghosting
        sys.stdout.write(f"\r{self.prefix} [{bar}] {current_step}/{self.total_steps} ({elapsed:.1f}s){suffix}\033[K")
        sys.stdout.flush()
        
        if current_step == self.total_steps:
            sys.stdout.write("\n")
            sys.stdout.flush()