# Retail Price & Sales Prediction AI 🤖

This project uses Machine Learning (XGBoost) to predict the **Optimal Recommended Price** and **Projected Sales Volume** for retail products. It is designed to handle:

- **Luxury Brands** (Rolex, Gucci, etc.)
- **Unbranded/Generic Items**
- **Bulk Processing** (Thousands of items at once)
- **Single Item Analysis**

---

## 📋 Requirements

To run this tool, you need **Python 3.8+** installed.

The project relies on the following libraries:

- `pandas` (Data manipulation)
- `numpy` (Math operations)
- `xgboost` (The Machine Learning model)
- `scikit-learn` (Metrics and data processing)

---

## ⚙️ Installation

1. **Unzip** the project folder.
2. Open your **Terminal** (Mac/Linux) or **Command Prompt/PowerShell** (Windows) inside the folder.
3. **Install Dependencies** (This is the Python equivalent of `npm i`):

```bash
pip install pandas numpy xgboost scikit-learn
train model:-
python train_model.py
single prediction:-
python predict.py --mode single
bulk production :-
python predict.py --mode bulk --file test_50.csv
```
