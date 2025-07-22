# ModelAnalysis.py
# This script analyzes performance of R1-AID
#Import libraries
from tqdm import tqdm
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, confusion_matrix, roc_curve, brier_score_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from datetime import datetime
from sklearn.calibration import calibration_curve

#Random Setup
SEED = 15
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
n_bootstrap = 1000
ci = 0.95

plt.rcParams.update({
    'font.size': 12,            # Base font size
    'axes.titlesize': 14,       # Title size of axes
    'axes.labelsize': 12,       # X and Y label size
    'xtick.labelsize': 10,      # X tick label size
    'ytick.labelsize': 10,      # Y tick label size
    'legend.fontsize': 11,      # Legend font size
    'figure.titlesize': 14      # Figure title size
})

def bootstrap_auc(y_true, y_prob, n_bootstraps=1000, ci=0.95):
    """
    Bootstrap confidence intervals for AUC.
    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bootstraps (int): Number of bootstrap samples.
        ci (float): Confidence interval level (default 0.95).
    Returns:
        tuple: Lower and upper bounds of the confidence interval for AUC.
    """
    rng = np.random.RandomState(SEED)
    bootstraps = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        bootstraps.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lower = np.percentile(bootstraps, (1-ci)/2*100)
    upper = np.percentile(bootstraps, (ci + (1-ci)/2)*100)
    return lower, upper


def bootstrap_auprc(y_true, y_prob, n_bootstraps=1000, ci=0.95):
    """
    Bootstrap confidence intervals for AUPRC.
    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bootstraps (int): Number of bootstrap samples.
        ci (float): Confidence interval level (default 0.95).
    Returns:
        tuple: Lower and upper bounds of the confidence interval for AUPRC.
    """
    rng = np.random.RandomState(SEED)
    bootstraps = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        bootstraps.append(average_precision_score(y_true[idx], y_prob[idx]))
    lower = np.percentile(bootstraps, (1-ci)/2*100)
    upper = np.percentile(bootstraps, (ci + (1-ci)/2)*100)
    return lower, upper

def calculate_confidence_intervals(scores, confidence_level=0.95):
    """
    Calculate confidence intervals from bootstrap scores
    Args:
        scores (array-like): Array of bootstrap scores.
        confidence_level (float): Confidence level for the interval (default 0.95).
    Returns:
        dict: Dictionary containing mean, lower CI, upper CI, and standard deviation of scores.
    """
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(scores, lower_percentile)
    ci_upper = np.percentile(scores, upper_percentile)
    mean_score = np.mean(scores)
    
    return {
        'mean': mean_score,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': np.std(scores)
    }

def bootstrap_metrics_at_threshold(y_true, y_prob, threshold, n_bootstrap=1000, ci=0.95):
    """
    Bootstrap confidence intervals for all metrics at a given threshold.
    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities for the positive class.
        threshold (float): Decision threshold to apply.
        n_bootstrap (int): Number of bootstrap samples.
        ci (float): Confidence interval level (default 0.95).
    Returns:
        dict: Dictionary containing mean, lower CI, and upper CI for each metric.
    """
    rng = np.random.RandomState(SEED)
    n = len(y_true)
    metrics_keys = list(calculate_metrics_at_threshold(y_true, y_prob, threshold).keys())
    # initialize storage
    store = {k: [] for k in metrics_keys if k != 'threshold'}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        true_boot = y_true[idx]
        prob_boot = y_prob[idx]

        # skip if only one class present
        if len(np.unique(true_boot)) < 2:
            continue

        m = calculate_metrics_at_threshold(true_boot, prob_boot, threshold)
        for k, v in m.items():
            if k != 'threshold':
                store[k].append(v)

    # compute CIs
    results = {}
    alpha = 1 - ci
    lo_p = 100 * (alpha / 2)
    hi_p = 100 * (1 - alpha / 2)
    for k, vals in store.items():
        arr = np.array(vals)
        results[k] = {
            'mean': np.nanmean(arr),
            'ci_lower': np.nanpercentile(arr, lo_p),
            'ci_upper': np.nanpercentile(arr, hi_p)
        }
    return results

def calculate_metrics_at_threshold(y_true, y_prob, threshold=0.5):
    """
    Calculate various metrics at a given threshold.
    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities for the positive class.
        threshold (float): Decision threshold to apply (default 0.5).   
    Returns:
        dict: Dictionary containing sensitivity, specificity, PPV, NPV, F1 score, accuracy, and Brier score.
    """

    # Binary predictions from probabilities
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Compute metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)

    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'accuracy': accuracy,
        'brier_score': brier
    }

def find_youden_threshold(y_true, y_prob):
    """
    Find the threshold that maximizes Youden's J = sensitivity + specificity - 1.
    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities for the positive class.
    Returns:
        tuple: Optimal threshold and a dictionary of metrics at that threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    idx = np.argmax(youden_j)
    thresh = thresholds[idx]
    metrics = calculate_metrics_at_threshold(y_true, y_prob, thresh)
    return thresh, metrics

#Load precomputed data
with open('cv_results.pkl', 'rb') as f:
    cv_results = pickle.load(f)

with open('cv-ensemble_preds.pkl', 'rb') as f:
    ensemble_preds = pickle.load(f)

with open('final_test_results.pkl', 'rb') as f:
    final_test_results = pickle.load(f)

with open('train_indices_main.pkl', 'rb') as f:
    train_indices_main = pickle.load(f)

with open('test_indices_main.pkl', 'rb') as f:
    test_indices_main = pickle.load(f)

with open('X.pkl', 'rb') as f:
    X_data = pickle.load(f)

with open('y.pkl', 'rb') as f:
    y_data = pickle.load(f)

#Obtain ensemble CV results
fold_trues = ensemble_preds['fold_trues']
fold_preds = ensemble_preds['fold_preds']
cv_y_true = np.concatenate(fold_trues)
cv_y_pred = np.concatenate(fold_preds)[:,1]
cv_auc = roc_auc_score(cv_y_true, cv_y_pred)
lo_cv, hi_cv = bootstrap_auc(cv_y_true, cv_y_pred)
print(f"Cross-validation AUC: {cv_auc:.4f} (95% CI: {lo_cv:.4f}-{hi_cv:.4f})")

fpr_cv, tpr_cv, thresholds_cv = roc_curve(cv_y_true, cv_y_pred)

n_bootstraps = 1000
ci = 0.95
rng = np.random.RandomState(SEED)
base_fpr = np.linspace(0, 1, 101)
tprs_boot_cv = []

for _ in range(n_bootstraps):
    idx = rng.randint(0, len(cv_y_true), len(cv_y_true))
    if len(np.unique(cv_y_true[idx])) < 2:
        continue
    fpr_bs, tpr_bs, _ = roc_curve(cv_y_true[idx], cv_y_pred[idx])
    # interpolate this bootstrap's TPR onto base_fpr
    tprs_boot_cv.append(np.interp(base_fpr, fpr_bs, tpr_bs))

tprs_boot_cv = np.vstack(tprs_boot_cv)
lower_perc = ((1 - ci) / 2) * 100
upper_perc = (ci + (1 - ci) / 2) * 100
tpr_lower_cv = np.percentile(tprs_boot_cv, lower_perc, axis=0)
tpr_upper_cv = np.percentile(tprs_boot_cv, upper_perc, axis=0)

#Obtain final test results
y_true = final_test_results['y_test_true']
y_prob = final_test_results['y_test_prob'][:,1]
test_auc = roc_auc_score(y_true, y_prob)
lo, hi = bootstrap_auc(y_true, y_prob)
print(f"Final test AUC: {test_auc:.4f} (95% CI: {lo:.4f}-{hi:.4f})")
fpr, tpr, thresholds = roc_curve(y_true, y_prob)

n_bootstraps = 1000
ci = 0.95
rng = np.random.RandomState(SEED)
base_fpr = np.linspace(0, 1, 101)
tprs_boot = []

for _ in range(n_bootstraps):
    idx = rng.randint(0, len(y_true), len(y_true))
    if len(np.unique(y_true[idx])) < 2:
        continue
    fpr_bs, tpr_bs, _ = roc_curve(y_true[idx], y_prob[idx])
    # interpolate this bootstrap's TPR onto base_fpr
    tprs_boot.append(np.interp(base_fpr, fpr_bs, tpr_bs))

tprs_boot = np.vstack(tprs_boot)
lower_perc = ((1 - ci) / 2) * 100
upper_perc = (ci + (1 - ci) / 2) * 100
tpr_lower = np.percentile(tprs_boot, lower_perc, axis=0)
tpr_upper = np.percentile(tprs_boot, upper_perc, axis=0)

#Create Figure 3A
#AUROC Curves: CV data under test data 
plt.figure(figsize=(6,6))
#Plot CV ROC
plt.plot(base_fpr, np.mean(tprs_boot_cv, axis=0), color='darkgray', lw=2, label=f"CV ROC: AUC = {cv_auc:.3f} (95% CI: {lo_cv:.3f}-{hi_cv:.3f})")
plt.fill_between(
    base_fpr, tpr_lower_cv, tpr_upper_cv,
    color='darkgray', alpha=0.4
)
#Plot Test ROC
plt.plot(base_fpr, np.mean(tprs_boot, axis=0), color='blue', lw=2, label=f"Test ROC: AUC = {test_auc:.3f} (95% CI: {lo:.3f}-{hi:.3f})")
plt.fill_between(
    base_fpr, tpr_lower, tpr_upper,
    color='lightblue', alpha=0.3
)
#Plot the diagonal line
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc-comparison.svg', format='svg')
plt.savefig('roc-comparison.png', dpi=1200)
plt.show()

#Create calibration curve
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
plt.figure()
plt.plot(mean_pred, frac_pos, marker='o', color='blue',lw=2)
plt.plot([0, 1], [0, 1], '--',color='gray')
plt.xlabel('Mean Predicted'); plt.ylabel('Fraction Positives')
plt.tight_layout()

#Calculate AUPRC for CV
auprc_cv = average_precision_score(cv_y_true, cv_y_pred)
lo_pr_cv, hi_pr_cv = bootstrap_auprc(cv_y_true, cv_y_pred)
print(f"Cross-validation AUPRC: {auprc_cv:.4f} (95% CI: {lo_pr_cv:.4f}-{hi_pr_cv:.4f})")

#Calculate AUPRC for final test
auprc_test = average_precision_score(y_true, y_prob)
lo_pr, hi_pr = bootstrap_auprc(y_true, y_prob)
print(f"Final test AUPRC: {auprc_test:.4f} (95% CI: {lo_pr:.4f}-{hi_pr:.4f})")

#Boostrap metrics at threshold 0.5
print("\nBootstrap CIs at threshold 0.5:")
ci_metrics = bootstrap_metrics_at_threshold(y_true, y_prob, threshold=0.5,
                                               n_bootstrap=n_bootstrap,
                                               ci=ci)

for k, d in ci_metrics.items():
    print(f"  {k}: mean={d['mean']:.4f}, CI=[{d['ci_lower']:.4f}-{d['ci_upper']:.4f}]")

# Youden's J optimal threshold metrics
# Find Youden's J threshold
youden_thresh, youden_metrics = find_youden_threshold(y_true, y_prob)
print(f"\nYouden's J optimal threshold: {youden_thresh:.4f}")
for k, v in youden_metrics.items():
    if k != 'threshold': print(f"  {k}: {v:.4f}")

# Bootstrap metrics at Youden threshold
ci_youden = bootstrap_metrics_at_threshold(y_true, y_prob,
                                            threshold=youden_thresh,
                                            n_bootstrap=n_bootstrap,
                                            ci=ci)
print("\nBootstrap CIs at Youden threshold:")
for k, d in ci_youden.items():
    print(f"  {k}: mean={d['mean']:.4f}, CI=[{d['ci_lower']:.4f}-{d['ci_upper']:.4f}]")
