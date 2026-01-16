# SARIMA-GRU: Time Series Forecasting

Hybrid deep learning model combining SARIMA + GRU for time series forecasting, optimized for water level prediction.

## 🚀 Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Train
```bash
python3 scripts/train.py --num_epochs 100 --batch_size 32
```

## 📁 Structure

```
├── src/sarima_gru/          # Core model
│   ├── model.py
│   ├── data.py
│   └── trainer.py
├── scripts/                 # Scripts
│   ├── train.py
│   ├── test.py
│   └── evaluate.py
├── dataset/                 # Data

```

## 📬 Data Availability

The dataset used in this study is not publicly available due to data-sharing restrictions.
Researchers interested in accessing the data for academic purposes may contact the authors via email:

📧 Email: pmduc2808@gmail.com

## 🎯 Commands

```bash
# Quick test
python3 scripts/train.py --num_epochs 10 --no_plot

# Standard
python3 scripts/train.py --num_epochs 100

# Full training
python3 scripts/train.py --num_epochs 500 --hidden_size 128

# Help
python3 scripts/train.py --help
```
---
**v1.0.0** | January 2026
