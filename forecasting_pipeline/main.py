import os
import torch
import matplotlib.pyplot as plt

from core_utils.config import parse_config
from core_utils.logger import get_logger
from forecasting_pipeline.config import ForecastingConfig
from forecasting_pipeline.dataset import build_loaders
from forecasting_pipeline.models import ForecastingLSTM, ForecastingGRU, TurningPointBiRNN
from forecasting_pipeline.trainer import ForecastingTrainer

logger = get_logger("ForecastingMain")

def get_optimizer(model: torch.nn.Module, config: ForecastingConfig) -> torch.optim.Optimizer:
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW
    }
    opt_class = optimizers.get(config.optimizer.lower(), torch.optim.AdamW)
    return opt_class(trainable_params, lr=config.lr, weight_decay=config.weight_decay)

def build_model(arch: str, num_features: int, config: ForecastingConfig) -> torch.nn.Module:
    arch_lower = arch.lower()
    if arch_lower == "lstm":
        return ForecastingLSTM(num_features, config)
    if arch_lower == "gru":
        return ForecastingGRU(num_features, config)
    if arch_lower in ("bilstm", "bigru"):
        cell_type = "lstm" if arch_lower == "bilstm" else "gru"
        return TurningPointBiRNN(num_features, config, cell_type=cell_type)
    raise ValueError(f"Unsupported network architecture: {arch}")

def run_experiment(config: ForecastingConfig, target_type: str, arch: str, label: str) -> dict:
    logger.info(f"Executing Experiment {label.upper()} | Target: {target_type} | Architecture: {arch}")
    
    train_loader, val_loader, test_loader, meta = build_loaders(config, target_type)
    
    # Model instantiation is bound strictly onto the mapped device context
    model = build_model(arch, meta["num_features"], config)
    optimizer = get_optimizer(model, config)
    
    # Core trainer orchestration mapping config parameters
    trainer = ForecastingTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        task_type=target_type,
        pos_weight=meta["pos_weight"]
    )
    
    is_classification = (target_type == "turning")
    target_metric = config.select_metric if is_classification else "loss"
    maximize = is_classification and target_metric in ("ap", "f1")
        
    save_path = f"{config.model_dir}/{label}_{arch}.pt"
    
    if config.mode in ("train", "both"):
        trainer.train(
            target_metric_name=target_metric, 
            maximize_metric=maximize, 
            patience=config.patience, 
            save_path=save_path
        )
    
    test_metrics = {}
    if config.mode in ("test", "both"):
        logger.info(f"Evaluating model performance on test partition: {label}_{arch}")
        trainer._load_checkpoint(save_path)
        
        # Explicitly re-bind trainer context to test split for deployment check
        trainer.val_loader = test_loader
        test_metrics = trainer._evaluate_epoch(epoch=0)
        logger.info(f"Final Test Metrics [{label}_{arch}]: {test_metrics}")

    return {
        "test_metrics": test_metrics,
        "history": getattr(trainer, 'history', [])
    }

def generate_report_and_plots(results: dict, history: dict):
    # 1. Print beautifully formatted table
    print("\n" + "="*85)
    print(f"{'Experiment':<12} | {'Architecture':<15} | {'Test Loss':<10} | {'Test Acc':<10} | {'Test AP':<10} | {'Test F1':<10}")
    print("-" * 85)
    for exp_key, metrics in results.items():
        if not metrics:
            continue
        parts = exp_key.split('_')
        exp = parts[0].upper()
        arch = parts[1].upper()
        loss = f"{metrics.get('loss', 0):.4f}"
        acc = f"{metrics.get('accuracy', 0):.4f}"
        ap = f"{metrics.get('ap', 0):.4f}" if 'ap' in metrics else "N/A"
        f1 = f"{metrics.get('f1', 0):.4f}" if 'f1' in metrics else "N/A"
        print(f"{exp:<12} | {arch:<15} | {loss:<10} | {acc:<10} | {ap:<10} | {f1:<10}")
    print("="*85 + "\n")
    
    # 2. Automated Plot Generation
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: exp_b_vs_c_loss_curves.png
    plt.figure(figsize=(12, 6))
    for key, hist in history.items():
        if key.startswith('b_') or key.startswith('c_'):
            if not hist: continue
            epochs = [h['epoch'] for h in hist]
            train_loss = [h['train_loss'] for h in hist]
            val_loss = [h['val_loss'] for h in hist]
            parts = key.split('_')
            label = f"Exp {parts[0].upper()} {parts[1].upper()}"
            plt.plot(epochs, train_loss, linestyle='--', label=f"{label} Train", alpha=0.7)
            plt.plot(epochs, val_loss, linestyle='-', label=f"{label} Val", linewidth=2)
    plt.title("Experiment B vs C: Training and Validation Loss Curves", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "exp_b_vs_c_loss_curves.png"))
    plt.close()

    # Plot 2: exp_b_c_accuracy_comparison.png
    plt.figure(figsize=(10, 6))
    archs = ['lstm', 'gru']
    b_accs = [results.get(f"b_{a}", {}).get("accuracy", 0) for a in archs]
    c_accs = [results.get(f"c_{a}", {}).get("accuracy", 0) for a in archs]
    
    x = range(len(archs))
    width = 0.35
    plt.bar([i - width/2 for i in x], b_accs, width, label='Exp B (Exact Returns)', color='#3498db', alpha=0.8)
    plt.bar([i + width/2 for i in x], c_accs, width, label='Exp C (Rolling Avg)', color='#2ecc71', alpha=0.8)
    
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Final Test Accuracy: Experiment B vs Experiment C', fontsize=14)
    plt.xticks(x, [a.upper() for a in archs], fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "exp_b_c_accuracy_comparison.png"))
    plt.close()

    # Plot 3: exp_d_turning_point_metrics.png
    plt.figure(figsize=(10, 6))
    d_keys = [k for k in results.keys() if k.startswith('d_')]
    if d_keys:
        d_archs = [k.split('_')[1] for k in d_keys]
        aps = [results[k].get("ap", 0) for k in d_keys]
        f1s = [results[k].get("f1", 0) for k in d_keys]
        
        x_d = range(len(d_keys))
        plt.bar([i - width/2 for i in x_d], aps, width, label='Test AP', color='#9b59b6', alpha=0.8)
        plt.bar([i + width/2 for i in x_d], f1s, width, label='Test F1', color='#e74c3c', alpha=0.8)
        
        plt.ylabel('Score', fontsize=12)
        plt.title('Experiment D: Turning Point Detection Metrics (AP & F1)', fontsize=14)
        plt.xticks(x_d, [a.upper() for a in d_archs], fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "exp_d_turning_point_metrics.png"))
        plt.close()

def main():
    config = parse_config(ForecastingConfig, "Financial Forecasting Pipeline")
    torch.manual_seed(config.seed)
    
    # Normalize inputs to safely strip accidental "bi" prefixes for regression tasks
    raw_arch = config.arch.lower()
    if raw_arch == "both":
        base_archs = ["lstm", "gru"]
    else:
        # If user provides "bilstm" directly, normalize base regression to "lstm"
        base_archs = [raw_arch.replace("bi", "")]
    
    results = {}
    history = {}

    # Experiment B: Standard Regression (d-day ahead returns)
    if config.experiment in ("b", "all"):
        for arch in base_archs:
            res = run_experiment(config, "return", arch, "b")
            results[f"b_{arch}"] = res["test_metrics"]
            history[f"b_{arch}"] = res["history"]
            
    # Experiment C: Smoothed Regression (Weighted rolling average returns)
    if config.experiment in ("c", "all"):
        for arch in base_archs:
            res = run_experiment(config, "rolling", arch, "c")
            results[f"c_{arch}"] = res["test_metrics"]
            history[f"c_{arch}"] = res["history"]
            
    # Experiment D: Structural Turning Point Binary Classification
    if config.experiment in ("d", "all"):
        for arch in base_archs:
            # Map clean baseline sequences dynamically into bidirectional frameworks for class boundaries
            bi_arch = "bilstm" if arch == "lstm" else "bigru"
            res = run_experiment(config, "turning", bi_arch, "d")
            results[f"d_{bi_arch}"] = res["test_metrics"]
            history[f"d_{bi_arch}"] = res["history"]

    # Generate final reporting and visualization
    generate_report_and_plots(results, history)

if __name__ == "__main__":
    main()