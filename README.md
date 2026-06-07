# ChurnSight: Customer Churn Prediction & Retention Analytics

**Problem:** Telecom companies lose ~$1,500 per churned customer in replacement acquisition costs. Retention teams need to know *who* is at risk and *why* — before they leave.

**Solution:** ChurnSight is an end-to-end ML system that predicts individual customer churn probability (XGBoost, ROC-AUC 0.847), explains each prediction using SHAP feature attribution, and delivers tiered retention recommendations mapped to real business cost economics.

**Impact:** Business-cost-optimised threshold (0.45) saves $14,900 vs. default on test set — extrapolated to $740K+ annually on a 70K-customer base. Deployed on Streamlit Cloud.

## Technical Architecture
- **Model:** XGBoost with `scale_pos_weight=2.8` (SMOTE tested and dropped — see Modeling Decisions)
- **Explainability:** SHAP TreeExplainer — global importance + per-customer local reasoning
- **Threshold Tuning:** Cost-optimised (FN=$1,500 | FP=$200) with precision floor ≥ 0.50
- **Validation:** 5-fold stratified CV — Mean AUC 0.845 ± 0.012
- **Stack:** Python · XGBoost · SHAP · Streamlit · Plotly · scikit-learn · imbalanced-learn

## Key Findings
- **ContractRisk** and **ChargesPerTenureMonth** are the top two SHAP drivers — both engineered features, not raw columns
- Month-to-month customers churn at ~42% vs 3% for two-year contracts
- New customers (≤6 months tenure) represent the highest-risk churn window
- Customers with more services have lower churn due to switching cost effect

## Project Structure
```
churn-prediction/
├── data/
│   ├── telco_churn.csv                # raw dataset (IBM Telco)
│   ├── telco_churn_clean.csv          # engineered features
│   ├── X_test.csv / y_test.csv        # held-out test set
│   └── predictions_with_shap.csv      # scored + SHAP drivers
├── models/
│   ├── xgboost_model.pkl              # model + threshold + metrics
│   ├── scaler.pkl                     # StandardScaler for LR baseline
│   └── shap_values_test.npy           # precomputed SHAP values
├── notebooks/
│   ├── 01_eda_and_features.ipynb      # EDA + 6 engineered features
│   ├── 02_modeling.ipynb              # Baseline → XGBoost + threshold tuning
│   └── 03_shap_analysis.ipynb         # Global + local SHAP explainability
├── assets/                            # Generated chart images
├── app.py                             # Streamlit dashboard (3 pages)
└── requirements.txt
```

## Run Locally
```bash
pip install -r requirements.txt
# Run notebooks in order: 01 → 02 → 03
streamlit run app.py
```