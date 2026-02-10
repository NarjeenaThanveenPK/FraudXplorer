# FraudXplorer: Credit Card Fraud Detection on Highly Imbalanced Data using Supervised Machine Learning


## Overview
Machine learning system for detecting fraudulent credit card transactions using supervised learning models. The project focuses on handling extreme class imbalance and optimizing recall–precision trade-offs common in real-world financial fraud detection.
This project demonstrates an end-to-end machine learning workflow from data preprocessing and model evaluation to deployment via a web interface.


## Problem Statement
Credit card fraud causes billions in losses annually. This project builds a classification system to identify fraudulent transactions, helping financial institutions minimize losses while maintaining customer experience.

## Dataset
**Source:** PaySim Synthetic Financial Dataset ([Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1))

- The original dataset contains ~6.3 million transactions.
- Due to size constraints, a representative subset (400,000 rows) is used for model training and evaluation.
- The dataset exhibits severe class imbalance (~0.1% fraudulent transactions), reflecting real-world fraud scenarios.

> Note: The full dataset is not included in the repository. Users can download it from Kaggle and place it in the `data/` directory if required.

**Demo:** Small sample data included in `data/dashboard_sample1.csv` for testing the Streamlit app.

## Approach

### Data Preprocessing
- Handled class imbalance using SMOTE (Synthetic Minority Over-sampling)
- Normalized transaction amounts
- Encoded categorical transaction types
- Dropped identifying columns (nameOrig, nameDest)

### Models Tested
1. Logistic Regression (baseline)
2. Decision Tree
3. Random Forest
4. XGBoost

### Evaluation Metrics
Focused primarily on **Recall** and **Precision** due to cost asymmetry:
- False Negative (missed fraud) = High cost
- False Positive (false alarm) = Lower cost

## Results

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Logistic Regression | 1.00 | 0.86 | 0.92 |
| Decision Tree | 1.00 | 0.90 | 0.95 |
| Random Forest | 1.00 | 0.97 | 0.99 |
| **XGBoost** | **1.00** | **0.99** | **0.99** |

> Note: Metrics are reported on a held-out test set from a synthetic dataset. 
> Precision values appear high due to strong class separation in PaySim and threshold tuning.


**XGBoost selected as final model** due to excellent balance of precision and recall.

## Deployment
Built a Streamlit-based web interface for batch fraud prediction on uploaded transaction data:
- Upload transaction CSV
- Get instant fraud predictions on uploaded data
- View detected fraud cases and statistics

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `SMOTE` `Streamlit` `Matplotlib` `Seaborn`

## Installation & Usage
```bash
# Clone repository
git clone https://github.com/NarjeenaThanveenPK/FraudXplorer.git
cd FraudXplorer

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run src/fraudxplorer_app.py
```

## Project Structure
```
FraudXplorer/
├── notebooks/
│   ├── FraudXplorer_xgb.ipynb      # Main analysis notebook
│   └── SampledashB.ipynb           # Dashboard data preparation
├── data/
│   └── dashboard_sample1.csv       # Sample data for demo
├── src/
│   └── fraudxplorer_app.py         # Streamlit application for inference
├── models/
│   └── xgb_model.pkl               # Trained XGBoost model
├── README.md
└── requirements.txt
```

## Limitations & Future Work
- Trained on synthetic data (real-world patterns may differ)
- No concept drift handling (model may degrade over time)
- Could add: Real-time monitoring, threshold tuning interface, feature importance analysis

## Author
Narjeena Thanveen P K