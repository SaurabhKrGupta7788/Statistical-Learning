# Statistical Learning Internship @ IIT Madras

Welcome to the official repository for my Summer Internship (May–July 2025) at the **Department of Mathematics, IIT Madras**, under the esteemed guidance of **Prof. Neelesh S. Upadhye**.

This internship focuses on the practical and theoretical foundations of **Statistical Learning**, including data modeling, model evaluation, and optimization techniques using real-world datasets.

---

##  Internship Overview

- **Institute**: Indian Institute of Technology, Madras  
- **Mentor**: Prof. Neelesh S. Upadhye  
- **Duration**: 18 May 2025 – 18 July 2025  
- **Focus Areas**:  
  - Statistical Modeling  
  - Machine Learning  
  - Applied Regression  
  - Model Evaluation & Optimization  

- **Base Environment**: `iitm-stats-learning/Conda_Env`

---

## Reading Resources
- [An Introduction to Statistical Learning](https://www.statlearning.com/?utm_source=chatgpt.com)
- [Probabilistic Machine Learning](https://probml.github.io/pml-book/?utm_source=chatgpt.com)

---

## Tasks Completed

###  Week 1: Boston Housing Dataset
- Performed Exploratory Data Analysis (EDA)
- Built an Ordinary Least Squares (OLS) Regression Model to predict housing prices
- Implemented 10-Fold Cross-Validation for robust evaluation
- Visualized feature correlations, scatter plots, and residual diagnostics  
   Notebook: `Saurabh_Kr_Gupta_BOSTON_HOUSING_W1.ipynb`

---

###  Week 2: Bias-Variance Tradeoff & Model Tuning
- Investigated and visualized the bias-variance tradeoff
- Compared underfitting and overfitting models
- Introduced Polynomial Regression techniques
- Validated models using RMSE and R² metrics  
   Notebook: `Saurabh_Kr_Gupta_Heart_W2.ipynb`

---


##  Environment Setup

This project uses a custom Conda environment. To set it up locally:

```bash
git clone https://github.com/iitm-stats-learning/Conda_Env.git
cd Conda_Env
conda env create -f stats_learning.yml
conda activate stats-learning
jupyter notebook
