# Statistical Learning Internship @ IIT Madras

Welcome to the official repository for my Summer Internship (May–July 2025) at the **Department of Mathematics, IIT Madras**, under the esteemed guidance of **Prof. Neelesh S. Upadhye**.

This internship focuses on the practical and theoretical foundations of **Statistical Learning**, including data modeling, model evaluation, and optimization techniques using real-world datasets.

---

## Internship Overview

* **Institute**: Indian Institute of Technology, Madras

* **Mentor**: Prof. Neelesh S. Upadhye

* **Duration**: 18 May 2025 – 18 July 2025

* **Focus Areas**:

  * Statistical Modeling
  * Machine Learning
  * Applied Regression
  * Model Evaluation & Optimization

* **Base Environment**: `iitm-stats-learning/Conda_Env`

---

## Reading Resources

* [An Introduction to Statistical Learning](https://www.statlearning.com/?utm_source=chatgpt.com)
* [Probabilistic Machine Learning](https://probml.github.io/pml-book/?utm_source=chatgpt.com)

---

## Tasks Completed

### Week 1: Boston Housing Dataset

* Performed Exploratory Data Analysis (EDA)
* Built an Ordinary Least Squares (OLS) Regression Model to predict housing prices
* Implemented 10-Fold Cross-Validation for robust evaluation
* Visualized feature correlations, scatter plots, and residual diagnostics
  Notebook: `Saurabh_Kr_Gupta_BOSTON_HOUSING_W1.ipynb`

---

### Week 2: Bias-Variance Tradeoff & Model Tuning

* Investigated and visualized the bias-variance tradeoff
* Compared underfitting and overfitting models
* Introduced Polynomial Regression techniques
* Validated models using RMSE and R² metrics
  Notebook: `Saurabh_Kr_Gupta_Heart_W2.ipynb`

---

### Week 3: Model Selection & Regularization

* Built and evaluated **Ridge** and **Lasso** regression models
* Used **Elastic Net** concepts for balancing L1 and L2 penalties
* Explored **feature importance** under different regularization strategies
* Explained regularization effects using **Bias-Variance Tradeoff** theory
* Performed **cross-validation and grid search** for tuning lambda/alpha
  Notebook: `Linear_Model_Selection_Regulation_W3.pdf`
  Code: `Saurabh_Kr_Gupta_Ridge&Lasso_week3_2.ipynb`

---

### Week 4: Ensemble Methods & Non-linear Models

* Compared performance of **Random Forest** and **XGBoost** on classification task
* Visualized feature importance and decision boundaries
* Analyzed bias-variance and interpretability tradeoffs between tree-based models
* Summarized findings in a PDF comparing both methods
  Notebook: `Saurabh_Kr_Gupta_RndmForest_vs_Boosting_W4.ipynb`
  Summary: `Randomforest_VS_Boosting_summary.pdf`

---

### Week 5: Capstone Kick-Off

* All model comparison on one dataset
* Cleaned raw data using **custom data-cleaning module**
* Created structured pipeline for preprocessing and validation
* Designed evaluation notebooks comparing models based on performance
* Assigned capstone project roles and created a Kanban-based tracking plan
  Notebooks:

  * `Saurabh_Kr_Gupta_model_comparision_W5.ipynb`
  * `Saurabh_Kr_Gupta_DataCleaning_W5_2.ipynb`
  * `all_model.py`
  * `cleaning_module.py`

---

### Week 6: Bayesian Modeling

* Implemented **Bayesian Linear Regression** using analytical posterior sampling
* Built a from-scratch **Bayesian Logistic Regression** with variational inference
* Studied **MAP vs. Full Bayesian vs. MLE** approaches for predictive modeling
* Visualized **posterior distributions, uncertainty bands**, and decision boundaries
* Compared Bayesian and frequentist interpretations for model reliability
  Notebook: `Saurabh_Kr_Gupta_Bayesian_Regression_W6.ipynb`
  Summary: `Bayesian_Methods_W6.pdf`

---

### Week 7: Model Comparison & Interpretability Analysis

* Conducted **comparative analysis** across all implemented models
* Investigated **bias-variance tradeoff** deeply through multiple test scenarios
* Measured and visualized **model interpretability vs. predictive power**
* Created final **evaluation matrix and performance summary charts**
* Documented key findings and recommendations for model selection strategy
  Notebook: `Saurabh_Kr_Gupta_Final_Model_Review_W7.ipynb`
  Report: `Model_Comparison_Summary_W7.pdf`

---

## Environment Setup

This project uses a custom Conda environment. To set it up locally:

```bash
git clone https://github.com/iitm-stats-learning/Conda_Env.git
cd Conda_Env
conda env create -f stats_learning.yml
conda activate stats-learning
jupyter notebook
```
