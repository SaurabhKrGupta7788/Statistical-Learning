# ===================================================Library Import =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import subplots
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import random
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis



# ==================================Dataset study plot (pass without values)===================================================
def data_plot(data , X, y):
    
    
    # 2.  Univariate Distribution + Skewness & Kurtosis
    print(" Skewness & Kurtosis of features:")
    for col in X.columns:
        s = skew(X[col])
        k = kurtosis(X[col])
        print(f"{col:<20} | Skew: {s:.2f} | Kurtosis: {k:.2f}")
        plt.figure(figsize=(6, 3))
        sns.histplot(X[col], kde=True)
        plt.title(f'Distribution of {col} (Skew: {s:.2f}, Kurtosis: {k:.2f})')
        plt.tight_layout()
        plt.show()

    # 3.  Feature vs Target (charges)
    print("\n Relationship with Target (charges):")
    for col in X.columns:
        plt.figure(figsize=(6, 3))
        sns.scatterplot(x=X[col], y=y)
        plt.title(f'{col} vs Charges')
        plt.tight_layout()
        plt.show()

    # 4.  PCA - High-dimensional Data Spread
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(' PCA: Data Spread in 2D')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Charges')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5. Correlation matrix
    cor_rel = data.corr()
    k,l = subplots(figsize = (28,15))
    sns.heatmap(cor_rel, annot= True, cmap='coolwarm' )
    l.set_title('Correlation matrix')
    
    
    
    # ========================================= Function plot (y vs y_pres and y_pred vs residual) ===================================
    
def plot(y, y_pred):
    residuals = y - y_pred

    plt.figure(figsize=(20,6))
    plt.scatter(y, y_pred, color='blue', edgecolor='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.grid(True)
    plt.show()

    # Residual plot
    plt.figure(figsize=(20,6))
    plt.scatter(y_pred, residuals, color='purple', edgecolor='k')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.show()
    
    # =========================================== Ordinary Least Square =============================================================
    

def corr(x, y):
    x_avg = np.mean(x)
    y_avg = np.mean(y)
    cor_num = np.sum((x - x_avg) * (y - y_avg))
    cor_den = np.sqrt(np.sum((x - x_avg)**2) * np.sum((y - y_avg)**2))
    return cor_num / cor_den

def run_multiple_linear_regression(X, y,X_test = 0, y_test = 0, feature_names=None):
    """
    Performs multiple linear regression using normal equation method and generates evaluation metrics + plots.

    Parameters:
        X (ndarray): n x p predictor matrix (without intercept)
        y (ndarray): n x 1 response vector
        feature_names (list): Optional, list of feature names for plotting
        plot (bool): If True, show plots

    Returns:
        dict: containing beta, y_pred, RSE, R², correlation, F-stat
    """
    n, p = X.shape
    X_design = np.hstack((np.ones((n, 1)), X))  # Add intercept term
    beta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    y_pred = X_design @ beta

    # Compute statistics
    residuals = y - y_pred
    tss = np.sum((y - np.mean(y)) ** 2)
    rss = np.sum((y - y_pred) ** 2)
    RSE = np.sqrt(rss / (n - p - 1))
    R_squared = 1 - rss / tss
    r = corr(y, y_pred)

    # F-statistic
    F_stat = ((tss - rss) / p) / (rss / (n - p - 1))

    # # Print results
    print("\n--- Train Evaluation ---")

    #print("Regression Coefficients (Beta):", beta)
    print("\n")
    print("Residual Standard Error (RSE):", RSE)
    print("R-squared:", R_squared)
    print("Correlation between actual and predicted:", r)
    print("R-squared vs. Correlation squared:", R_squared, "vs", r ** 2)
    if round(F_stat) == 1:
        print("H0: F-statistic is ~1, model may not be significant.")
    else:
        print("H1: F-statistic =", F_stat, "→ likely significant.")
        
    if X_test is not None and y_test is not None:
        test_result = test_evaluation_using_beta(beta, X_test, y_test)

    
    plot(y, y_pred)


    return {
        "beta": beta,
        "y_pred": y_pred,
        "RSE": RSE,
        "R_squared": R_squared,
        "correlation": r,
        "F_stat": F_stat,
        "residuals": residuals
    }

def test_evaluation_using_beta(beta, X_test, y_test):
    n, p = X_test.shape
    X_test_design = np.hstack((np.ones((n, 1)), X_test))  # Add intercept
    y_pred_test = X_test_design @ beta

    tss = np.sum((y_test - np.mean(y_test)) ** 2)
    rss = np.sum((y_test - y_pred_test) ** 2)
    RSE = np.sqrt(rss / (n - p - 1))
    R_squared = 1 - rss / tss
    r = corr(y_test, y_pred_test)

    print("\n--- Test Evaluation ---")
    print("Residual Standard Error (RSE):", RSE)
    print("R-squared:", R_squared)
    print("Correlation between actual and predicted:", r)
    print("R-squared vs. Correlation squared:", R_squared, "vs", r ** 2)

    return {
        "RSE": RSE,
        "R_squared": R_squared,
        "correlation": r,
        "y_pred": y_pred_test,
        "residuals": y_test - y_pred_test
    }


# ====================================================Logistic Regression ============================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def log_likelihood(beta, x, y):
    z = np.dot(x, beta)
    h = sigmoid(z)
    return -np.sum(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))  # for numerical stability

def hessian(beta, x):
    z = sigmoid(np.dot(x, beta))
    w = np.diag(z * (1 - z))
    return -np.dot(np.dot(x.T, w), x)

def run_logistic_regression_analysis(x, y):
    x_copy = x.copy()

    # Add intercept
    x = np.column_stack((np.ones(len(x)), x))
    init_beta = np.zeros(x.shape[1])

    # Optimize using BFGS
    result = minimize(log_likelihood, init_beta, args=(x, y), method='BFGS')
    b = result.x
    # print("Coefficients:")
    # for i, coeff in enumerate(b):
    #     print(f"  b{i} = {coeff:.4f}")

    # Predictions
    y_pred_proba = sigmoid(np.dot(x, b))
    y_pred_class = (y_pred_proba >= 0.5).astype(int)

    # Accuracy
    accuracy = accuracy_score(y, y_pred_class)
    print(f"\n Accuracy: {accuracy * 100:.2f}%")

    # Standard Errors
    H = hessian(b, x)
    cov_matrix = np.linalg.inv(-H)
    std_err = np.sqrt(np.diag(cov_matrix))

    # Print standard error and Confidence Intervals
    # print("\nStandard Errors & Confidence Intervals:")
    # conf_intervals = {}
    # for i in range(len(b)):
    #     ci_low = b[i] - 2 * std_err[i]
    #     ci_up = b[i] + 2 * std_err[i]
    #     conf_intervals[f"b{i}"] = (ci_low, ci_up)
    #     print(f"  SE_b{i} = {std_err[i]:.4f},  CI_b{i} = [{ci_low:.4f}, {ci_up:.4f}]")

    # Z-statistics and P-values
    # print("\n Z-statistics & P-values:")
    # z_stats = {}
    # p_values = {}
    # for i in range(len(b)):
    #     z = b[i] / std_err[i]
    #     p = 2 * norm.sf(abs(z))
    #     z_stats[f"b{i}"] = z
    #     p_values[f"b{i}"] = p
    #     print(f"  z_b{i} = {z:.4f},  p_b{i} = {p:.4f} → {'Reject H0' if p < 0.05 else 'Fail to reject H0'}")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred_class)
    print("\n Confusion Matrix:\n", cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve & AUC
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'LogReg (AUC = {auc_score:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        'coefficients': b,
        'standard_errors': std_err,
        # 'conf_intervals': conf_intervals,
        # 'z_stats': z_stats,
        # 'p_values': p_values,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'auc_score': auc_score,
        'y_pred_prob': y_pred_proba,
        'y_pred_class': y_pred_class
    }

def test_logistic(beta, x_test, y_test):
    # Add intercept
    x_test = np.column_stack((np.ones(len(x_test)), x_test))

    # Predict probabilities and class
    y_pred_proba = sigmoid(np.dot(x_test, beta))
    y_pred_class = (y_pred_proba >= 0.5).astype(int)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_class)
    print("\nTest Confusion Matrix:\n", cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Test Confusion Matrix')
    plt.show()

    # ROC Curve & AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Test AUC = {auc_score:.2f}', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'auc_score': auc_score,
        'y_pred_prob': y_pred_proba,
        'y_pred_class': y_pred_class
    }



# ==================================================== Ridge Regression (L2 Regression) ==============================================




def train_f(x, y, lembda):
    r, c = x.shape
    identity = np.eye(c)
    return np.linalg.inv(x.T @ x + lembda * identity) @ x.T @ y

def test_f(x, y, lembda, b_train):
    r, c = x.shape
    preds = x @ b_train
    return (1/r) * np.sum((preds - y)**2)

def test_x_pred(x, y, lembda, b_train):
    r, c = x.shape
    preds = x @ b_train
    test_r2 = r2_score(y, preds)
    test_mse = (1/r) * np.sum((preds - y)**2)
    print(f' Test R2: {test_r2:.4f}')
    print(f' Test MSE: {test_mse:.4f}')
    return

def prediction_evaluation(data, x,y, lembda = np.linspace(0, 100, 9000)):

    col = data.columns

    n = len(x)

    # Standardizing predictors
    x_mean = np.mean(x, axis=0)
    x_stdiv = np.std(x, axis=0)
    x_std = (x - x_mean) / x_stdiv

    #finding the k-part data
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf.get_n_splits(x_std)
    kf.split(x_std)

    splits = []
    #x_test,x_train,y_test,y_train = 0,0,0,0
    for train_index, test_index in kf.split(x_std):
        x_train , x_test = x_std[train_index] , x_std[test_index]
        y_train , y_test = y[train_index] , y[test_index]
        splits.append((x_train, y_train, x_test, y_test))

    (x_train_fold1, y_train_fold1, x_test_fold1, y_test_fold1) = splits[0]
    (x_train_fold2, y_train_fold2, x_test_fold2, y_test_fold2) = splits[1]
    (x_train_fold3, y_train_fold3, x_test_fold3, y_test_fold3) = splits[2]
    (x_train_fold4, y_train_fold4, x_test_fold4, y_test_fold4) = splits[3]
    (x_train_fold5, y_train_fold5, x_test_fold5, y_test_fold5) = splits[4]

    train_error = []
    test_error = []

    #train test on split 5 dataset
    for i in lembda:
        train_error_fold = []
        test_error_fold = []

        b = (train_f(x_train_fold1, y_train_fold1, i))
        train_error_fold.append(test_f(x_train_fold1, y_train_fold1, i, b))
        test_error_fold.append(test_f(x_test_fold1, y_test_fold1, i, b))

        b = (train_f(x_train_fold2, y_train_fold2, i))
        train_error_fold.append(test_f(x_train_fold2, y_train_fold2, i, b))
        test_error_fold.append(test_f(x_test_fold2, y_test_fold2, i, b))


        b = (train_f(x_train_fold3, y_train_fold3, i))
        train_error_fold.append(test_f(x_train_fold3, y_train_fold3, i, b))
        test_error_fold.append(test_f(x_test_fold3, y_test_fold3,i, b))

        b = (train_f(x_train_fold4, y_train_fold4, i))
        train_error_fold.append(test_f(x_train_fold4, y_train_fold4, i, b))
        test_error_fold.append(test_f(x_test_fold4, y_test_fold4, i, b))

        b = (train_f(x_train_fold5, y_train_fold5, i))
        train_error_fold.append(test_f(x_train_fold5, y_train_fold5, i, b))
        test_error_fold.append(test_f(x_test_fold5, y_test_fold5, i, b))

        # calculating average of error for each lambda
        train_error.append(np.average(train_error_fold))
        test_error.append(np.average(test_error_fold))

    best_index = np.argmin(test_error)
    best_lambda = lembda[best_index]
    print(f"The best lambda is : {best_lambda:.4f}")

    # Train on full dataset with best lambda
    b_ridge = train_f(x_std, y, best_lambda)  # coefficients without intercept

    # Un-standardize coefficients
    b_aft_std = b_ridge / x_stdiv
    b0 = np.mean(y) - np.sum(b_aft_std * x_mean)

    # b_full includes intercept
    b_full = np.insert(b_aft_std, 0, b0)

    # Compute MSE on full dataset - must add intercept column to x_std
    x_full = np.c_[np.ones(n), x]

    y_pred_full = x_full @ b_full
    y_pred_ridge = y_pred_full

    final_mse = np.mean((y - y_pred_full)**2)
    print(f'Final training MSE is : {final_mse:.4f}\n')



    # # Print coefficients
    # print(f"Intercept (b0): {b0:.4f}")
    # for i in range(len(b_aft_std)):
    #     print(f"Coefficient of {col[i]}: {b_aft_std[i]:.4f}")

    # Calculate RSS and TSS using full standardized data with intercept
    RSS = np.sum((y - y_pred_full)**2)
    TSS = np.sum((y - np.mean(y))**2)
    # print(f'Total Sum of Squares (TSS) is : {TSS:.4f}')
    # print(f'Residual Sum of Squares (RSS) is : {RSS:.4f}')

    # Mallow's Cp
    d = len(b_aft_std)  # number of predictors without intercept

    # OLS fit for sigma^2 estimation
    b_ols = np.linalg.inv(x_full.T @ x_full) @ (x_full.T @ y)
    residual_ols = y - x_full @ b_ols
    rss_full = np.sum(residual_ols**2)
    sigma_sq = rss_full / (n - d - 1)
    
    
    train_r2 = r2_score(y, y_pred_ridge)
    print(f'Train R^2 : {train_r2:.4f}')

    train_mse = mean_squared_error(y, y_pred_ridge)
    print(f'Train MSE : {train_mse:.4f}')

    
    
    Cp = (1 / n) * (RSS + 2 * d * sigma_sq)
    # print(f'CP is : {Cp:.4f}')

    # Adjusted R^2
    adj_R_sqr = 1 - ((RSS * (n - 1)) / (TSS * (n - d - 1)))
    print(f'Adjusted R^2 is : {adj_R_sqr:.4f}')

    # AIC and BIC
    aic = n * np.log(RSS / n) + 2 * d
    bic = n * np.log(RSS / n) + d * np.log(n)
    print(f'AIC is : {aic:.4f}')
    print(f'BIC is : {bic:.4f}')

    return b_full ,y_pred_full, best_lambda 





def ridge_model(data ,X_train , y_train , X_test, y_test, lembda):
    
    b_full , y_train_pre , best_lembda = prediction_evaluation(data, X_train, y_train,lembda)
    x_test = np.column_stack((np.ones(X_test.shape[0]), X_test))
    
    test_x_pred(x_test, y_test, best_lembda, b_full)
    
    col = [f"x{i}" for i in range(X_train.shape[1])]
    
    coefs = []
    for l in lembda:
        b = train_f(X_train, y_train, l)
        coefs.append(b)
    coefs = np.array(coefs)

    plt.figure(figsize=(20,6))
    for i in range(coefs.shape[1]):
        plt.plot(lembda, coefs[:, i], label=col[i])
    plt.xlabel("Lambda")
    plt.ylabel("Coefficients")
    plt.title("Ridge: Coefficient Paths vs Lambda")
    plt.legend(loc='best', bbox_to_anchor=(1,1))
    plt.grid(True)
    plt.show()
    
    
    plot(y_train, y_train_pre)
    return b_full
# ============================================ Lasso Regression (L1 Regularization) ==================================================



# Soft Thresholding Function
def soft_threshold(rho, lam):
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0.0

# Lasso via Coordinate Descent
def lasso_coordinate_descent(X, y, lam, tol=1e-4, max_iter=1000):
    n, p = X.shape
    beta = np.zeros(p)

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            X_j = X[:, j]
            y_pred = X @ beta
            residual = y - y_pred + beta[j] * X_j
            rho_j = np.dot(X_j, residual)
            beta[j] = soft_threshold(rho_j / n, lam) / (np.dot(X_j, X_j) / n)
        if np.sum(np.abs(beta - beta_old)) < tol:
            break
    return beta

# Adjusted R² Function
def adjusted_r2_score(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def lasso(x_full,y_full, lambda_values = np.logspace(-2, 3, 200)):
    


    x,x_test , y, y_test = train_test_split(x_full,y_full, test_size=0.2, random_state=42 )
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_standardized = (x - x_mean) / x_std

    
    # Center y
    y_mean = np.mean(y)
    y_centre = y - y_mean

    # Cross-validation settings
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    avg_cv_errors = []
    coefs = []

    # Cross-validation loop
    for lam in lambda_values:
        cv_errors = []
        coef_path = []

        for train_idx, val_idx in kf.split(x_standardized):
            X_train, X_val = x_standardized[train_idx], x_standardized[val_idx]
            y_train_cen = y_centre[train_idx]
            y_val = y[val_idx]

            beta_lasso = lasso_coordinate_descent(X_train, y_train_cen, lam)
            coef_path.append(beta_lasso)

            y_pred = X_val @ beta_lasso
            mse = mean_squared_error(y_val, y_pred + y_mean)
            cv_errors.append(mse)

        avg_cv_errors.append(np.mean(cv_errors))
        coefs.append(np.mean(coef_path, axis=0))

    coefs = np.array(coefs)
    best_lambda = lambda_values[np.argmin(avg_cv_errors)]
    print(f"Best Lambda Value: {best_lambda:.4f}")

    # Final model training
    final_beta = lasso_coordinate_descent(x_standardized, y_centre, best_lambda)
    final_beta_lasso = final_beta / x_std
    beta_0 = y_mean - np.dot(final_beta_lasso, x_mean)
    beta_full = np.insert(final_beta_lasso, 0, beta_0)

    # Predictions on original scale
    X_full = np.column_stack((np.ones(x.shape[0]), x))
    y_pred_lasso = X_full @ beta_full

    # test prediction
    x_test_full = np.column_stack((np.ones(x_test.shape[0]), x_test))
    y_pre_test = x_test_full @ beta_full
    
    
    # Metrics
    rss = np.sum((y - y_pred_lasso) ** 2)
    rss_test = np.sum((y_test - y_pre_test)**2)
    tss = np.sum((y - y_mean) ** 2)
    n = len(y)
    p = x.shape[1]
    sigma_squared = rss / (n - p - 1)

    cp = (rss + 2 * p * sigma_squared) / n
    aic = cp
    bic = (rss + np.log(n) * p * sigma_squared) / n
    mse = mean_squared_error(y, y_pred_lasso)
    mse_test = mean_squared_error(y_test,y_pre_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred_lasso)
    r2_test = r2_score(y_test, y_pre_test)
    adj_r2 = adjusted_r2_score(y, y_pred_lasso, p)

    col_names = [f"x{i}" for i in range(X_train.shape[1])]


    
    

    

    print("\nLasso Results:")
    print(f"Train RSS: {rss:.2f}")
    print(f"Test RSS: {rss_test:.2f}")
    print(f"Estimated Variance σ²: {sigma_squared:.2f}")
    print(f"Mallow's Cp: {cp:.2f}")
    print(f"AIC: {aic:.2f}")
    print(f"BIC: {bic:.2f}")
    print(f"Train MSE: {mse:.2f}")
    print(f"Test MSE: {mse_test:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² train: {r2:.4f}")
    print(f"R² test: {r2_test:.4f}")
    print(f"Adjusted R²: {adj_r2:.4f}")
    
    
    
    # Plot: Lambda vs MSE
    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, avg_cv_errors, color='blue')
    plt.xscale('log')
    plt.xlabel("Lambda Values")
    plt.ylabel("Mean Squared Error")
    plt.title("Lambda vs MSE")
    plt.scatter(best_lambda, avg_cv_errors[np.argmin(avg_cv_errors)], color='red', s=60, label="Best Lambda")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot: Coefficients vs Lambda
    plt.figure(figsize=(10, 8))
    for i in range(coefs.shape[1]):
        plt.plot(lambda_values, coefs[:, i], label=col_names[i])
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("Coefficient Value")
    plt.title("Lasso Coefficients vs Lambda")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plot(y, y_pred_lasso)
    
    return beta_full
# ======================================================== Regression Tree =========================================================


#  RSS function
def rss(y, y_pred):
    return np.sum((y - y_pred) ** 2)

#  Midpoint split generation
def s_for_x(x):
    s = []
    unique_vals = np.sort(np.unique(x))
    for i in range(len(unique_vals) - 1):
        midpoint = (unique_vals[i] + unique_vals[i + 1]) / 2
        s.append(midpoint)
    return s

#  Recursive Binary Split
def rec_binary_split(x, y, x_full):
    s_xj = []
    rss_xj = []

    for j in range(x.shape[1]):
        xj = x[:, j]
        rss_er = []
        s = s_for_x(xj)

        for i in s:
            R1 = xj < i
            R2 = xj >= i

            if np.sum(R1) == 0 or np.sum(R2) == 0:
                continue  # skip invalid split

            y_left = y[R1]
            y_right = y[R2]

            rss_val = rss(y_left, np.mean(y_left)) + rss(y_right, np.mean(y_right))
            rss_er.append(rss_val)

        if len(rss_er) == 0:
            continue  # skip if all splits are invalid

        min_rss_indx = np.argmin(rss_er)
        min_rss = rss_er[min_rss_indx]
        threshold = s[min_rss_indx]

        s_xj.append(threshold)
        rss_xj.append(min_rss)

    if len(rss_xj) == 0:
        return 0, np.mean(x[:, 0]), x_full, x_full, y, y  # fallback split

    rss_xj_indx = np.argmin(rss_xj)
    s_best = s_xj[rss_xj_indx]
    best_feature_index = rss_xj_indx

    x_column = x_full[:, best_feature_index]
    x_full_R1 = x_full[x_column < s_best]
    x_full_R2 = x_full[x_column >= s_best]
    y_R1 = y[x_column < s_best]
    y_R2 = y[x_column >= s_best]

    return best_feature_index, s_best, x_full_R1, x_full_R2, y_R1, y_R2

#  Tree building
def build_tree(x, y, depth=0, max_depth=5):
    if depth >= max_depth or len(y) <= 1:
        return {'leaf_value': np.mean(y)}

    feature_index, threshold, x_left, x_right, y_left, y_right = rec_binary_split(x, y, x)

    left_subtree = build_tree(x_left, y_left, depth + 1, max_depth)
    right_subtree = build_tree(x_right, y_right, depth + 1, max_depth)

    return {
        'feature_index': feature_index,
        'threshold': threshold,
        'left': left_subtree,
        'right': right_subtree
    }

#  Leaf check
def is_leaf(node):
    return 'leaf_value' in node

#  Leaf count
def count_leaves(tree):
    if is_leaf(tree):
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

#  RSS calculation for whole tree
def calculate_rss(tree, x_true, y_true):
    if is_leaf(tree):
        return np.sum((y_true - tree['leaf_value']) ** 2)

    feature = tree['feature_index']
    threshold = tree['threshold']

    left_mask = x_true[:, feature] < threshold
    right_mask = x_true[:, feature] >= threshold

    return calculate_rss(tree['left'], x_true[left_mask], y_true[left_mask]) + calculate_rss(tree['right'], x_true[right_mask], y_true[right_mask])

#  Pruning using cost-complexity
def cost_complexity_prune(tree, x_true, y_true, alpha):
    if is_leaf(tree):
        return tree

    feature = tree['feature_index']
    threshold = tree['threshold']

    left_mask = x_true[:, feature] < threshold
    right_mask = x_true[:, feature] >= threshold

    left_pruned = cost_complexity_prune(tree['left'], x_true[left_mask], y_true[left_mask], alpha)
    right_pruned = cost_complexity_prune(tree['right'], x_true[right_mask], y_true[right_mask], alpha)

    new_tree = {
        'feature_index': feature,
        'threshold': threshold,
        'left': left_pruned,
        'right': right_pruned
    }

    rss_subtree = calculate_rss(new_tree, x_true, y_true)
    rss_leaf = rss(y_true, np.mean(y_true))
    leaves = count_leaves(new_tree)

    cost_leaf = rss_leaf + alpha * 1
    cost_subtree = rss_subtree + alpha * leaves

    if cost_leaf <= cost_subtree:
        return {'leaf_value': np.mean(y_true)}
    else:
        return new_tree


#  Alpha selection using cross-validation
def select_best_alpha_by_cv(X, y, alphas, k=5, max_depth=3):
    y = pd.Series(y)  # ensure we can use iloc
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    avg_rss_per_alpha = []

    for alpha in alphas:
        rss_list = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            tree = build_tree(X_train, y_train.values, max_depth=max_depth)
            pruned_tree = cost_complexity_prune(tree, X_train, y_train.values, alpha)

            rss_val = calculate_rss(pruned_tree, X_val, y_val.values)
            rss_list.append(rss_val)

        avg_rss = np.mean(rss_list)
        avg_rss_per_alpha.append(avg_rss)

    best_alpha_index = np.argmin(avg_rss_per_alpha)
    best_alpha = alphas[best_alpha_index]
    return best_alpha, avg_rss_per_alpha

#  Final pruned tree
def get_best_subtree(tree, X_train, y_train, alpha):
    return cost_complexity_prune(tree, X_train, y_train, alpha)


def predict_single(tree, x):
    """Predict output for a single sample using the tree."""
    if is_leaf(tree):
        return tree['leaf_value']

    feature = tree['feature_index']
    threshold = tree['threshold']

    if x[feature] < threshold:
        return predict_single(tree['left'], x)
    else:
        return predict_single(tree['right'], x)

def predict_tree(tree, X):
    predictions = []
    for i in range(len(X)):
        x_row = X[i]
        pred = predict_single(tree, x_row)
        predictions.append(pred)
    return np.array(predictions)

def training_mse(tree, X_train, y_train):
    y_pred = predict_tree(tree, X_train)
    return np.mean((y_train - y_pred) ** 2)


def r2_score_custom(tree, X, y_true):
    y_pred = predict_tree(tree, X)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def tree_model(X, y):
    # Step 1: Build full tree
    tree = build_tree(X, y)

    # Step 2: Select best alpha via CV
    alphas = np.linspace(0.00001, 20, 200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_alpha, rss_list = select_best_alpha_by_cv(X_train, y_train, alphas)

    # Step 3: Prune tree using best alpha
    best_tree = get_best_subtree(tree, X_train, y_train, best_alpha)

    # Step 4: Predictions
    # Predictions
    train_preds = predict_tree(best_tree, X_train)
    test_preds = predict_tree(best_tree, X_test)

    # Evaluation
    train_mse = mean_squared_error(y_train, train_preds)
    test_mse = mean_squared_error(y_test, test_preds)

    train_r2 = r2_score_custom(best_tree, X_train, y_train)
    test_r2 = r2_score_custom(best_tree, X_test, y_test)

    n,p = X_train.shape
    adj_r2_tree = adjusted_r2_score(y_train, train_preds,p )


    # Output
    print("\nFinal Model Evaluation Report")
    print("───────────────────────────────────")
    print(f'Best Alpha    : {best_alpha:.4f}')
    print(f"Train MSE     : {train_mse:.4f}")
    print(f"Test MSE      : {test_mse:.4f}")
    print(f"Train R² Score: {train_r2:.4f}")
    print(f"Test R² Score : {test_r2:.4f}")
    print(f" Adjusted R^2 : {adj_r2_tree:.4f}")

    plot(y_train, train_preds)

# =================================================== Random Forest ==============================================================


# Add this function if you don't have it for adjusted R²
def adjusted_r2_score(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

def pick_random(lst):
    random.seed(50)
    k = random.randint(1, len(lst))
    return random.sample(lst, k)

def random_forest(x, y, B):
    # 1️ Validate inputs and handle NaNs
    print("NaN in x:", x.isna().sum().sum())
    print("NaN in y:", y.isna().sum())
    
    if x.isna().sum().sum() > 0 or y.isna().sum() > 0:
        x = x.fillna(x.mean())
        y = y.fillna(y.mean())
        print("Warning: NaNs found and filled with mean")

    # 2️ Split data
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    # 3️ Ensemble storage
    train_preds_all = []
    test_preds_all = []

    # 4️ Tree building loop
    for i in range(B):
        # Random feature subset
        features = pick_random(X_train.columns.tolist())
        X_sub = X_train[features]
        
        # Bootstrap sampling
        X_boot, y_boot = resample(
            X_sub, y_train, 
            replace=True, 
            random_state=42 + i,  # Different seed per tree
            stratify=None
        )
        
        # Train tree
        tree = build_tree(X_boot.values, y_boot.values)
        
        # Get predictions
        train_pred = predict_tree(tree, X_sub.values)
        test_pred = predict_tree(tree, X_test[features].values)
        
        # Handle NaN in predictions
        if np.isnan(train_pred).any():
            train_pred = np.nan_to_num(train_pred, nan=np.nanmean(train_pred))
        if np.isnan(test_pred).any():
            test_pred = np.nan_to_num(test_pred, nan=np.nanmean(test_pred))
        
        # Store predictions
        train_preds_all.append(train_pred)
        test_preds_all.append(test_pred)

    # 5️ Average predictions
    train_pred_avg = np.mean(train_preds_all, axis=0)
    test_pred_avg = np.mean(test_preds_all, axis=0)
    
    # 6️ Ensure no NaNs remain
    assert not np.isnan(train_pred_avg).any(), "NaNs in final train predictions"
    assert not np.isnan(test_pred_avg).any(), "NaNs in final test predictions"

    # 7️ Evaluation metrics
    train_MSE = mean_squared_error(y_train, train_pred_avg)
    test_MSE = mean_squared_error(y_test, test_pred_avg)
    train_R2 = r2_score(y_train, train_pred_avg)
    test_R2 = r2_score(y_test, test_pred_avg)
    n, p = X_train.shape
    adj_r2_rf = adjusted_r2_score(y_train, train_pred_avg, p)

    # 8️ Results
    print("\n=== Random Forest Results ===")
    print(f"Trees: {B}, Features: {p}")
    print(f"Train MSE: {train_MSE:.4f}, Test MSE: {test_MSE:.4f}")
    print(f"Train R²: {train_R2:.4f}, Test R²: {test_R2:.4f}")
    print(f"Adjusted R²: {adj_r2_rf:.4f}")

    # 9️ Plotting (assuming you have a plot function)
    plot(y_train, train_pred_avg)
    
    return {
        "model": "random_forest",
        "train_mse": train_MSE,
        "test_mse": test_MSE,
        "train_r2": train_R2,
        "test_r2": test_R2,
        "adj_r2": adj_r2_rf
    }

# ===================================================== Gradient Boosting =========================================================



# # 2️ Boosting Function
# def boosting(X_train, y_train, X_test,y_test, n_estimators, learning_rate):
#     models = []
#     train_preds_all = []
#     test_preds_all = []

#     # Step 1: Initialize predictions to 0
#     y_pred_train = np.zeros_like(y_train)
#     y_pred_test = np.zeros(len(X_test))

#     # Step 2: Initialize residuals
#     residual = y_train.copy()
    
#     train_MSE_list = []
#     test_MSE_list = []
#     train_R2_list = []
#     test_R2_list = []


#     for i in range(n_estimators):
#         # 2(a): Fit tree on residuals
#         tree = build_tree(X_train, residual)

#         # Predict current tree output
#         pred_train = predict_tree(tree, X_train)
#         pred_test = predict_tree(tree, X_test)

#         # 2(b): Update predictions
#         y_pred_train += learning_rate * pred_train
#         y_pred_test += learning_rate * pred_test

#         # 2(c): Update residuals
#         residual -= learning_rate * pred_train

#         # Save current predictions
#         train_preds_all.append(np.copy(y_pred_train))
#         test_preds_all.append(np.copy(y_pred_test))

#         models.append(tree)
#         train_MSE_list.append(mean_squared_error(y_train, y_pred_train))
#         test_MSE_list.append(mean_squared_error(y_test, y_pred_test))

#         train_R2_list.append(r2_score(y_train, y_pred_train))
#         test_R2_list.append(r2_score(y_test, y_pred_test))
        
#     # 4️ Evaluate final predictions
#     train_pred_final = train_preds_all[-1]
#     test_pred_final = test_preds_all[-1]


#     train_MSE = mean_squared_error(y_train, train_pred_final)
#     test_MSE = mean_squared_error(y_test, test_pred_final)
#     train_R2 = r2_score(y_train, train_pred_final)
#     test_R2 = r2_score(y_test, test_pred_final)

#     print(" Final Train MSE:", train_MSE)
#     print(" Final Test MSE:", test_MSE)
#     print(" Final Train R²:", train_R2)
#     print(" Final Test R²:", test_R2)
#     n,p = X_train.shape
#     adj_r2_boost = adjusted_r2_score(y_train, train_pred_final,p )
#     print(" Adjusted R^2 :", adj_r2_boost)
#     plot(y_train, train_pred_final)
    
#     return 

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def xgboost_regressor(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=3):
    # 1. Ensure inputs are numpy arrays
    X_train = np.array(X_train) if not hasattr(X_train, 'values') else X_train.values
    X_test  = np.array(X_test)  if not hasattr(X_test,  'values') else X_test.values
    y_train = np.array(y_train).ravel() if hasattr(y_train, 'values') else np.array(y_train).ravel()
    y_test  = np.array(y_test).ravel()  if hasattr(y_test,  'values') else np.array(y_test).ravel()

    # 2. Handle missing values by imputing column means
    if np.isnan(X_train).any():
        # Compute column means (ignoring NaNs)
        col_means = np.nanmean(X_train, axis=0)
        # Find indices where NaNs
        inds = np.where(np.isnan(X_train))
        # Replace NaNs with corresponding column means
        X_train[inds] = np.take(col_means, inds[1])
    if np.isnan(X_test).any():
        col_means = np.nanmean(X_test, axis=0)
        inds = np.where(np.isnan(X_test))
        X_test[inds] = np.take(col_means, inds[1])
    if np.isnan(y_train).any():
        # Replace NaN targets with their mean
        y_train = np.nan_to_num(y_train, nan=np.nanmean(y_train))
    if np.isnan(y_test).any():
        y_test = np.nan_to_num(y_test, nan=np.nanmean(y_test))

    # 3. Initialize predictions and residuals
    y_pred_train = np.zeros_like(y_train, dtype=float)
    y_pred_test  = np.zeros(len(X_test),   dtype=float)
    residual = y_train.copy()  # initial residuals

    # 4. Boosting iterations: fit trees on residuals
    models = []
    for i in range(n_estimators):
        # Fit a regression tree to current residuals
        tree = build_tree(X_train, residual, max_depth=max_depth)
        models.append(tree)

        # Predict with this tree
        pred_train = predict_tree(tree, X_train)
        pred_test  = predict_tree(tree, X_test)

        # Update predictions with learning rate
        y_pred_train += learning_rate * pred_train
        y_pred_test  += learning_rate * pred_test

        # Update residuals
        residual -= learning_rate * pred_train

        # (Optional) Check for any new NaNs (just in case) and fix them
        if np.isnan(y_pred_train).any():
            mean_train = np.nanmean(y_pred_train)
            y_pred_train = np.nan_to_num(y_pred_train, nan=mean_train)
        if np.isnan(y_pred_test).any():
            mean_test = np.nanmean(y_pred_test)
            y_pred_test = np.nan_to_num(y_pred_test, nan=mean_test)

    # 5. Final evaluation metrics
    train_MSE = mean_squared_error(y_train, y_pred_train)
    test_MSE  = mean_squared_error(y_test,  y_pred_test)
    train_R2  = r2_score(y_train, y_pred_train)
    test_R2   = r2_score(y_test,  y_pred_test)
    n, p = X_train.shape
    # Adjusted R^2 helper (from all_model code)
    def adjusted_r2(y_true, y_pred, p):
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    train_adj_R2 = adjusted_r2(y_train, y_pred_train, p)
    test_adj_R2  = adjusted_r2(y_test,  y_pred_test, p)

    # Print results
    print(f"Final Train MSE: {train_MSE:.4f}")
    print(f"Final Test MSE:  {test_MSE:.4f}")
    print(f"Final Train R²:  {train_R2:.4f}")
    print(f"Final Test R²:   {test_R2:.4f}")
    print(f"Train Adj R²:    {train_adj_R2:.4f}")
    print(f"Test Adj R²:     {test_adj_R2:.4f}")

    return {
        "model":      "xgboost_regressor",
        "train_mse":  train_MSE,
        "test_mse":   test_MSE,
        "train_r2":   train_R2,
        "test_r2":    test_R2,
        "adj_r2":     train_adj_R2
    }

# ===========================Bayesian Linear Regression =========================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# Helper Plot Function
# ------------------------------
def plot_bayesian(true, pred, title="Prediction vs True"):
    plt.figure(figsize=(8, 4))
    plt.plot(true.values if hasattr(true, 'values') else true, label="True", color='red')
    plt.plot(pred, label="Prediction", linestyle='dashed')
    plt.title(title)
    plt.legend()
    plt.show()

# ------------------------------
# TRAINING FUNCTION
# ------------------------------
def train_bayesian_linear_regression(X_train, y_train, sigma2=0.01, tau2=1.0, n_samples=500):
    d = X_train.shape[1]
    I = np.eye(d)

    # Posterior mean and covariance
    Sigma_post = np.linalg.inv((1 / sigma2) * X_train.T @ X_train + (1 / tau2) * I)
    mu_post = (1 / sigma2) * Sigma_post @ X_train.T @ y_train

    # Sample w from posterior
    w_samples = np.random.multivariate_normal(mu_post.ravel(), Sigma_post, size=n_samples)
    y_pred_samples = np.array([X_train @ w for w in w_samples])

    y_pred_mean = y_pred_samples.mean(axis=0)
    y_pred_std  = y_pred_samples.std(axis=0)

    # Metrics
    train_mse = mean_squared_error(y_train, y_pred_mean)
    train_r2 = r2_score(y_train, y_pred_mean)

    print("\n--- Training Results (Bayesian Linear Regression) ---")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Train R² : {train_r2:.4f}")

    plot_bayesian(y_train, y_pred_mean, title="Train: True vs Predicted")

    return mu_post, Sigma_post

# ------------------------------
# TESTING FUNCTION
# ------------------------------
def test_bayesian_linear_regression(X_test, y_test, mu_post, Sigma_post, sigma2=0.01, n_samples=500):
    w_samples = np.random.multivariate_normal(mu_post.ravel(), Sigma_post, size=n_samples)
    y_pred_samples = np.array([X_test @ w for w in w_samples])

    y_pred_mean = y_pred_samples.mean(axis=0)
    y_pred_std = y_pred_samples.std(axis=0)

    test_mse = mean_squared_error(y_test, y_pred_mean)
    test_r2 = r2_score(y_test, y_pred_mean)

    print("\n--- Testing Results (Bayesian Linear Regression) ---")
    print(f"Test MSE : {test_mse:.4f}")
    print(f"Test R²  : {test_r2:.4f}")

    plot_bayesian(y_test, y_pred_mean, title="Test: True vs Predicted")

    # Visualize uncertainty (sample 100)
    sample_idx = np.random.choice(len(X_test), size=100, replace=False)
    X_sample = X_test[sample_idx]
    y_sample = y_test[sample_idx] if isinstance(y_test, np.ndarray) else y_test.iloc[sample_idx]
    y_sample_pred = X_sample @ mu_post
    y_sample_std = np.sqrt(np.array([sigma2 + x.T @ Sigma_post @ x for x in X_sample]))

    plt.figure(figsize=(10, 4))
    plt.errorbar(range(len(y_sample)), y_sample_pred, yerr=y_sample_std, fmt='o', label='Prediction ± std')
    plt.plot(range(len(y_sample)), y_sample, 'r-', label='True')
    plt.title("Sampled Test Predictions with Uncertainty")
    plt.legend()
    plt.show()


#========================== calling nature ==============================================

## Data for plotting


# X = insurance.drop('charges', axis=1)
# y = insurance['charges']
# data_plot(insurance, X, y)



## Data Arrangment for calling models

# selected_features = ['age', 'smoker_yes', 'bmi', 'region_southeast' , 'children' , 'sex_male' , 'region_northwest' , 'region_southwest' ]
# X = insurance[selected_features].values
# y = insurance['charges'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Linear Model :
# train_result = run_multiple_linear_regression(X_train, y_train,X_test, y_test, feature_names=selected_features)

## Ridge Regression :
# lembda=  np.linspace(0, 100, 9000) 
# ridge_model(insurance ,X_train , y_train , X_test, y_test, lembda)

## Lasso :
# lmda = np.linspace(-2,3,200)
# lasso(X_train, y_train, lmda)

## Regression Tree :
# tree_model(X,y)

## Random Forest :
# x_r = insurance[['age', 'smoker_yes', 'bmi', 'region_southeast' , 'children' , 'sex_male' , 'region_northwest' , 'region_southwest']]
# random_forest(x_r, y, 100)

## Gradinet Boosting :
# boosting(X_train, y_train, X_test,y_test, n_estimators = 500, learning_rate= 0.01)

## Bayesian Linear regression
# 1. Train model
#mu_post, Sigma_post = train_bayesian_linear_regression(X_train, y_train)
# 2. Evaluate on test data
#test_bayesian_linear_regression(X_test, y_test, mu_post, Sigma_post)
