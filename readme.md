 Product Intelligence Engine (PIE) 🤖
This project is an enterprise-grade unified AI platform for strategic retail decision-making. It integrates pricing, demand forecasting, and customer sentiment analysis into a single dashboard.
+1

The platform is designed to handle:
Intelligent Cold-Start Pricing: Pricing brand-new products with zero historical data.
Precision Demand Forecasting: Predicting sales volume using State-of-the-Art Gradient Boosting.
High-Scale Sentiment Analysis: Processing up to 1.6 Million tweets to capture real-time market pulse.
---

## 📋 Requirements

To run this tool, you need **Python 3.8+** installed.

The project relies on the following libraries:

- `pandas` (Data manipulation)
- `numpy` (Math operations)
- `xgboost` (The Machine Learning model)
- `scikit-learn` (Metrics and data processing)
- nltk (Natural Language Processing toolkit)
- streamlit (Interactive Frontend dashboard)

---

## ⚙️ Installation

1. **Unzip** the project folder.
2. Open your **Terminal** (Mac/Linux) or **Command Prompt/PowerShell** (Windows) inside the folder.
3. **Install Dependencies** (This is the Python equivalent of `npm i`):

```bash
pip install pandas numpy xgboost scikit-learn nltk streamlit
train model:-
python train_model.py
single prediction:-
python predict.py --mode single
bulk production :-
python predict.py --mode bulk --file test_50.csv
train the Sentiment Model & generate the Vectorizer:-
python Twitter_sentiment_analysis.ipynb
run the full Dashboard:-
streamlit run app.py
```
🏗️ Project Architecture
Module 1 (Pricing): Attribute-Weighted Heuristic Engine.
Module 2 (Forecasting): XGBoost with Temporal Feature Engineering.
Module 3 (Sentiment): Optimized NLP Pipeline (PorterStemmer & TF-IDF).
