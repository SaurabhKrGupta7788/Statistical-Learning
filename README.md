# Statistical Learning Internship @ IIT Madras

Official repository for the Summer Internship (May–July 2025) at the **Department of Mathematics, IIT Madras**, under the guidance of **Prof. Neelesh S. Upadhye**. This project focuses on the practical and theoretical foundations of Statistical Learning, covering data modeling, evaluation, and optimization via a structured pipeline.

## Architecture Overview

```text
Raw Datasets (Boston Housing, Heart Disease, etc.)
    │
    ▼
┌───────────────────────────────────────┐
│  Data Preprocessing & Cleaning        │  ← Custom cleaning_module.py
└──────────┬────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────┐
│  Statistical Modeling & ML Pipeline   │
│  - Week 1: OLS Regression             │
│  - Week 2: Polynomial Regression      │
│  - Week 3: Ridge / Lasso / ElasticNet │
│  - Week 4: RandomForest / XGBoost     │
│  - Week 6: Bayesian Linear & Logistic │
│  - Week 7: PCA & Clustering (K-Means) │
└──────────┬────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────┐
│  Model Evaluation & Cross-Validation  │ ← Grid Search, K-Fold CV, MAP vs MLE
└──────────────┬────────────────────────┘
               ▼
        System Output
   (Visualizations, RMSE/R2 Metrics, 
    Feature Importance, PDF Summaries)
```

## System Output

The workflow outputs Jupyter notebooks containing mathematical proofs, code implementations, and visualization artifacts:

| Phase | Output Type | Description |
|---|---|---|
| Exploratory | Plots/EDA | Scatter plots, correlation matrices, and residual diagnostics |
| Modeling | Metrics | RMSE, R², Accuracy scores across folds |
| Inference | Parameter Dists | Bayesian posterior distributions and Bias-Variance tradeoffs |
| Summaries | PDF Reports | Weekly digests explaining theoretical underpinnings |

## Directory Structure

```text
Statistical-Learning/
├── Datasets/                   # Raw and processed datasets (Boston, Heart, etc.)
├── Notebooks/                  # Weekly Jupyter notebooks for modeling & analysis
│   ├── Saurabh_Kr_Gupta_BOSTON_HOUSING_W1.ipynb
│   ├── Saurabh_Kr_Gupta_Heart_W2.ipynb
│   ├── Saurabh_Kr_Gupta_Ridge&Lasso_week3_2.ipynb
│   ├── Saurabh_Kr_Gupta_RndmForest_vs_Boosting_W4.ipynb
│   ├── Saurabh_Kr_Gupta_model_comparision_W5.ipynb
│   └── (Additional weekly notebooks...)
├── all_model.py                # Centralized modeling pipeline script
└── cleaning_module.py          # Custom data-cleaning module
```

## How to Run

### 1. Prerequisites
- Python 3.9+
- Jupyter Notebook / Lab
- `conda` for environment management

### 2. Install Dependencies

```bash
# Create base environment
conda create -n iitm-stats-learning python=3.10 -y
conda activate iitm-stats-learning

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn xgboost jupyter
```

### 3. Run the Application
This repository consists of analytical notebooks rather than a single application.

```bash
# Launch Jupyter environment
jupyter notebook
```

### 4. Usage
1. Open the `Notebooks/` directory in Jupyter.
2. Execute the notebooks sequentially (from W1 to W7) to follow the curriculum.
3. Review the custom modules (`cleaning_module.py` and `all_model.py`) to understand the underlying data processing and modeling abstractions.

## Key Design Decisions

1. **Modular Code Extraction**: In Week 5, redundant preprocessing and modeling logic across notebooks was abstracted into `cleaning_module.py` and `all_model.py` to adhere to DRY (Don't Repeat Yourself) principles.
2. **Progressive Complexity**: The architecture moves from simple parametric models (OLS) to regularized models (Lasso/Ridge), into non-linear ensembles (XGBoost), and finally fully probabilistic (Bayesian) approaches, providing a complete structural learning curve.
3. **Rigorous Evaluation Setup**: Every model is evaluated using K-Fold Cross Validation and Bias-Variance analysis to ensure true generalization rather than just in-sample overfitting.
