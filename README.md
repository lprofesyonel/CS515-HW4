```markdown
# CS515 Deep Learning — Homework 4: Sequence Modeling

This repository contains a comprehensive, production-grade implementation for Sabanci University's CS515 Deep Learning Homework 4. The project is partitioned into two self-contained neural pipeline environments, leveraging reusable infrastructure modules from `core_utils`.


```

main/
├── core_utils/               # Shared operational infrastructure
│   ├── config.py             # Dynamic CLI argument parser mapping
│   ├── logger.py             # Isolated logging handlers & CLI progress telemetry
│   └── trainer.py            # Abstract baseline optimization & checkpoint controller
├── forecasting_pipeline/     # Part 1 — Deep Recurrent Financial Forecasting Pipeline
│   ├── config.py             # Hyperparameter dataclasses and parsing boundaries
│   ├── data.py               # Live market ingestion (yfinance) & windowing primitives
│   ├── models.py             # Structural implementations of LSTM, GRU, and BiRNN
│   ├── train.py              # Convergence routines optimizing MSE and Cross-Entropy loss
│   ├── test.py               # Multi-horizon statistical backtesting & evaluation metrics
│   ├── main.py               # Orchestration gateway for Experiments B, C, and D
│   └── checkpoints/          # Local serialized state binaries (*.pt)
└── comm_system/              # Part 2 — Learned Transformer Feedback Communication Protocol
│   ├── config.py             # Configuration parameters for channel joint optimization
│   ├── models.py             # Post-LN Transformer blocks, Autoencoder TX/RX layers
│   ├── train.py              # End-to-end stochastic block optimization loops
│   ├── test.py               # Boundary evaluation routines (SNR sweeps, SER/BLER trackers)
│   └── main.py               # Execution entry point for the joint communication system
├── requirements.txt          # Dependencies for the entire project
└── README.md                 # Project documentation
```

## Part 1: Deep Recurrent Financial Forecasting

This subsystem deploys Deep Recurrent Neural Networks (LSTMs, GRUs, and Bidirectional RNNs) to model non-linear temporal dependencies across quantitative equity returns.

## Part 2: Learned Interactive Communication Protocol

An end-to-end parametric communication transceiver optimized over stochastic channel state distributions. The pipeline builds a learned feedback loop utilizing standard Post-Layer Normalization (Post-LN) Transformer blocks to code and decode message blocks under continuous power constraints.