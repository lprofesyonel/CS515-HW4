import os
import torch
import json
import matplotlib.pyplot as plt
from core_utils.config import parse_config
from core_utils.logger import get_logger
from comm_system.config import CommConfig
from comm_system.models import FeedbackTransceiver
from comm_system.trainer import EndToEndCommTrainer

logger = get_logger("CommMain")

def main():
    # Parse configuration parameters
    config = parse_config(CommConfig, "Learned Feedback Communication Pipeline")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    logger.info(
        f"Starting Transceiver Pipeline | Communication Rounds (T) = {config.t_rounds} | "
        f"Channel Noise Variance (sigma^2) = {config.noise_var}"
    )
    
    # Initialize the joint transmitter-receiver model
    model = FeedbackTransceiver(config)
    
    # Set up AdamW optimizer for trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    
    # Initialize the end-to-end communication trainer
    trainer = EndToEndCommTrainer(
        model=model,
        optimizer=optimizer,
        config=config
    )
    
    save_path = os.path.join(config.model_dir, "comm_system_best.pt")
    
    # Explicitly define the plots directory
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Execution Block: Joint Training Loop
    if config.mode in ("train", "both"):
        logger.info("Starting end-to-end autoencoder joint training...")
        trainer.train(
            target_metric_name="bler", 
            maximize_metric=False, 
            save_path=save_path
        )
        
        # 1. BLER Convergence Curve Plot
        history = trainer.history
        if history:
            epochs = [h['epoch'] for h in history]
            val_blers = [h['val_bler'] for h in history]
            
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, val_blers, color='#2980b9', linewidth=2, marker='o', markersize=4)
            plt.yscale('log')
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Block Error Rate (BLER)', fontsize=12)
            plt.title('BLER Convergence Curve', fontsize=14)
            plt.grid(True, which='both', linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(plots_dir, "bler_convergence_curve.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logger.info(f"Convergence plot saved to {plot_path}")

    # Execution Block: Evaluation over multiple SNR thresholds
    if config.mode in ("test", "both"):
        logger.info("Starting multi-SNR performance evaluation sweep...")
        trainer._load_checkpoint(save_path)
        results = trainer.evaluate_snr_sweep()
        
        json_path = os.path.join(plots_dir, "snr_sweep_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved raw sweep results to {json_path}")
    
        try:
            snr_axis = sorted([float(k) for k in results.keys()])
            bler_axis = [results[s]["bler"] for s in snr_axis]
            ser_axis = [results[s]["ser"] for s in snr_axis]
            
            plt.figure(figsize=(9, 6))
            plt.plot(snr_axis, bler_axis, marker='s', color='crimson', linestyle='-', linewidth=2, label='Block Error Rate (BLER)')
            plt.plot(snr_axis, ser_axis, marker='o', color='navy', linestyle='--', linewidth=1.5, label='Symbol Error Rate (SER)')
            
            plt.yscale('log') 
            plt.xlabel('SNR (dB)', fontsize=12)
            plt.ylabel('Error Rate (Log Scale)', fontsize=12)
            plt.title('Transceiver Performance over AWGN Channel', fontsize=14)
            plt.grid(True, which="both", linestyle=":", alpha=0.6)
            plt.legend(fontsize=11)
            
            plot_path = os.path.join(plots_dir, "snr_vs_error_rate.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Performance plot successfully saved to {plot_path}")
            
        except ImportError:
            logger.warning("matplotlib found missing. Skipping performance plot generation.")
            
        logger.info("Multi-SNR evaluation sweep completed successfully.")

    # ---------------------------------------------------------
    # FINAL AUTOMATED REPORTING & VISUALIZATION
    # ---------------------------------------------------------
    logger.info("Generating automated transmission power check...")
    
    # 2. Track transmission power using a forward hook
    power_per_round = []
    def tx_hook(module, args, output):
        # output is x_t normalized with shape (batch_size, seq_len)
        power_per_round.append((output ** 2).mean().item())

    # Attach hook to transmitter module
    handle = model.transmitter.register_forward_hook(tx_hook)
    
    # Forward pass to collect power averages across communication rounds
    with torch.no_grad():
        test_device = next(model.parameters()).device
        test_symbols = torch.randint(0, config.alphabet, (1000, config.n_symbols), device=test_device)
        model(test_symbols) # Simulate a pass to capture x_t via the hook
        
    handle.remove()
    
    # Plot average power over rounds
    rounds = list(range(1, config.t_rounds + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, power_per_round[:config.t_rounds], color='#d35400', marker='D', linewidth=2, linestyle='-')
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Average Power Limit (1.0)')
    plt.ylim(0, max(1.5, max(power_per_round[:config.t_rounds]) * 1.2))
    plt.xlabel('Communication Round (t)', fontsize=12)
    plt.ylabel(r'Average Transmit Power $\mathbb{E}||x^{(t)}||^2$', fontsize=12)
    plt.title('Transmission Power Verification', fontsize=14)
    plt.xticks(rounds)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    power_plot_path = os.path.join(plots_dir, "transmission_power_check.png")
    plt.savefig(power_plot_path, dpi=300)
    plt.close()
    logger.info(f"Power verification plot saved to {power_plot_path}")
    
    # 3. Generate Console Summary Block
    if config.mode in ("test", "both") and 'results' in locals() and results:
        final_bler = list(results.values())[-1]["bler"]
    elif config.mode in ("train", "both") and trainer.history:
        final_bler = trainer.history[-1]["val_bler"]
    else:
        final_bler = float('nan')
        
    status = "Converged" if trainer.epochs_without_improvement < getattr(config, 'patience', 10) else "Terminated (Early Stopped)"

    print("\n" + "="*50)
    print(" "*10 + "PERFORMANCE SUMMARY REPORT")
    print("="*50)
    print(f" Final BLER achieved : {final_bler:.5f}")
    print(f" Training status     : {status}")
    print(f" Model saved to      : {save_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()