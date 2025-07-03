# MCST-Mamba: Multi-Channel Spatio-Temporal Mamba for Traffic Prediction


This repository contains the official implementation of **MCST-Mamba**, a novel traffic prediction model that simultaneously forecasts multiple traffic features (speed, flow, occupancy) using Mamba state space models. This work extends the [LibCity](https://github.com/LibCity/Bigscity-LibCity) benchmarking framework.

## 🚀 Overview

MCST-Mamba addresses the challenge of predicting multiple interrelated traffic measurements by:

- **Multi-Channel Forecasting**: Simultaneously models all traffic features instead of requiring separate models
- **Dual Mamba Architecture**: Separate blocks for temporal sequences and spatial sensor interactions
- **Adaptive Embeddings**: Rich spatio-temporal representations for improved pattern learning
- **Efficient Design**: Lower parameter count while maintaining strong predictive performance

## 📊 Quick Start

### 1. Installation

```bash
git clone https://github.com/mombash/MCST-Mamba.git
cd Bigscity-LibCity
pip install -r requirements.txt
```

### 2. Train MCST-Mamba

```bash
# Basic usage with PEMSD8 dataset
python run_model.py --task traffic_state_pred --model MCSTMamba --dataset PEMSD8

# Custom parameters
python run_model.py --task traffic_state_pred --model MCSTMamba --dataset PEMSD8 \
    --d_model 96 --d_state 32 --input_window 12 --output_window 12
```

## 🔧 Advanced Usage

### Training on Different Datasets

```bash
# Use with different datasets
python run_model.py --task traffic_state_pred --model MCSTMamba --dataset METR_LA
python run_model.py --task traffic_state_pred --model MCSTMamba --dataset PEMSD4
```

### Hyperparameter Tuning

```bash
# Experiment with different model sizes
python run_model.py --task traffic_state_pred --model MCSTMamba --dataset PEMSD8 \
    --d_model 64 --d_state 32 --hidden_dim 128
```

### Model Evaluation

Our custom evaluation tool supports both single-channel and multi-channel inference:

```bash
# Basic evaluation (all channels together)
python evaluate_trained_model.py \
  --model MCSTMamba \
  --dataset PEMSD8 \
  --model_dir MCSTMamba_PEMSD8_20241201_120000 \
  --epoch 95

# Evaluate each channel separately
python evaluate_trained_model.py \
  --model MCSTMamba \
  --dataset PEMSD8 \
  --model_dir MCSTMamba_PEMSD8_20241201_120000 \
  --epoch 95 \
  --evaluate_channels_separately
```


## 📁 Project Structure

```
├── libcity/model/traffic_speed_prediction/
│   └── MCSTMamba.py              # Main MCST-Mamba implementation
├── libcity/config/model/traffic_state_pred/
│   └── MCSTMamba.json            # Model configuration file
├── run_model.py                  # Main execution script
├── evaluate_trained_model.py     # Custom evaluation script
└── requirements.txt              # Dependencies
```

## 🤝 Acknowledgments

This work builds upon the excellent [LibCity](https://github.com/LibCity/Bigscity-LibCity) framework, a comprehensive benchmarking platform for traffic prediction. LibCity provides:

- Unified data processing pipeline
- Standardized evaluation metrics
- Extensive model repository
- Reproducible experimental framework

## 📚 Citation

**Paper and citation information will be added soon.**

For now, please cite the original LibCity framework:

```bibtex
@inproceedings{libcity,
  author = {Wang, Jingyuan and Jiang, Jiawei and Jiang, Wenjun and Li, Chao and Zhao, Wayne Xin},
  title = {LibCity: An Open Library for Traffic Prediction},
  booktitle = {Proceedings of the 29th International Conference on Advances in Geographic Information Systems},
  year = {2021}
}
```

## 🐛 Issues

- Report bugs and request features via [GitHub Issues](https://github.com/your-repo/issues)
- For questions about MCST-Mamba, open an issue with the `[MCST-Mamba]` tag

---