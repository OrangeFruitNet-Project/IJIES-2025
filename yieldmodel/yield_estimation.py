# yield_estimation.py
"""
Yield estimation utilities for OrangeFruitNet.

Functions:
 - area_scaling(tree_imgs_df): baseline canopy-area scaling (Eq. 6)
 - train_regression(...): KFold CV on linear regression (Eq. 9) with feature scaling;
                         returns cross-val metrics and final_model trained on full data.
 - predict_with_regression(model, df_features): predict per-tree yield using model.

Assumptions / Units:
 - A (canopy area) in m^2
 - D (mean fruit diameter) in m
 - rho (fruit density) in fruits/m^2
 - true_yield in kg (or consistent unit)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

def area_scaling(tree_imgs_df):
    """
    Baseline area-scaling yield estimate per tree as in Eq. (6).

    Args:
        tree_imgs_df (pd.DataFrame): rows must include:
            'tree_id', 'image_id', 'detected_count', 'img_area_m2', 'tree_area_m2'

    Returns:
        pd.DataFrame: columns ['tree_id', 'est_count'] with aggregated estimates per tree
    """
    req_cols = {'tree_id','detected_count','img_area_m2','tree_area_m2'}
    if not req_cols.issubset(set(tree_imgs_df.columns)):
        raise ValueError(f"Missing required columns: {req_cols - set(tree_imgs_df.columns)}")
    # compute scaled count per row
    df = tree_imgs_df.copy()
    df['scaled_count'] = (df['tree_area_m2'] / df['img_area_m2']) * df['detected_count']
    grouped = df.groupby('tree_id', as_index=False)['scaled_count'].sum()
    grouped = grouped.rename(columns={'scaled_count': 'est_count'})
    return grouped

def _safe_bias_percent(y_true, y_pred, eps=1e-8):
    mean_true = np.mean(y_true)
    if abs(mean_true) < eps:
        # fallback: absolute mean bias (not percent)
        return np.mean(y_pred - y_true)
    return (np.mean(y_pred - y_true) / mean_true) * 100.0

def train_regression(tree_features_df, target_col='true_yield', n_splits=5, random_state=42, do_bootstrap_ci=False, n_bootstrap=1000):
    """
    Train a regression model (LinearRegression) with KFold cross-validation.

    Args:
        tree_features_df (pd.DataFrame): must contain columns ['A','D','rho', target_col]
        target_col (str): name of ground truth yield column (kg)
        n_splits (int): CV folds
        random_state (int)
        do_bootstrap_ci (bool): compute bootstrap CI for R2 (costly)
        n_bootstrap (int)

    Returns:
        final_model: sklearn Pipeline (StandardScaler + LinearRegression) trained on full data
        cv_summary: dict with cross-val metrics: r2_mean, r2_sd, rmse_mean, rmse_sd, bias_mean, bias_sd, per_fold_metrics
    """
    req_cols = {'A','D','rho', target_col}
    if not req_cols.issubset(set(tree_features_df.columns)):
        raise ValueError(f"Missing required columns: {req_cols - set(tree_features_df.columns)}")

    X = tree_features_df[['A','D','rho']].values
    y = tree_features_df[target_col].values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])
        pipe.fit(Xtr, ytr)
        ypred = pipe.predict(Xte)
        r2 = r2_score(yte, ypred)
        rmse = mean_squared_error(yte, ypred, squared=False)
        bias = _safe_bias_percent(yte, ypred)
        fold_metrics.append({'fold': fold_idx, 'r2': r2, 'rmse': rmse, 'bias_percent': bias})

    # aggregate
    r2s = np.array([m['r2'] for m in fold_metrics])
    rmses = np.array([m['rmse'] for m in fold_metrics])
    biases = np.array([m['bias_percent'] for m in fold_metrics])

    cv_summary = {
        'n_folds': n_splits,
        'r2_mean': float(np.mean(r2s)),
        'r2_sd': float(np.std(r2s, ddof=1) if len(r2s) > 1 else 0.0),
        'rmse_mean': float(np.mean(rmses)),
        'rmse_sd': float(np.std(rmses, ddof=1) if len(rmses) > 1 else 0.0),
        'bias_mean': float(np.mean(biases)),
        'bias_sd': float(np.std(biases, ddof=1) if len(biases) > 1 else 0.0),
        'per_fold': fold_metrics
    }

    # Final model trained on all data
    final_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
    ])
    final_pipe.fit(X, y)

    # Optional bootstrap CI for r2_mean (expensive)
    if do_bootstrap_ci:
        try:
            bs_r2 = []
            rng = np.random.RandomState(random_state)
            n = len(X)
            for _ in range(n_bootstrap):
                idx = rng.randint(0, n, size=n)
                Xb, yb = X[idx], y[idx]
                pipe_b = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
                pipe_b.fit(Xb, yb)
                ypred_b = pipe_b.predict(Xb)
                bs_r2.append(r2_score(yb, ypred_b))
            cv_summary['r2_bootstrap_ci'] = [float(np.percentile(bs_r2, 2.5)), float(np.percentile(bs_r2, 97.5))]
        except Exception as e:
            warnings.warn(f"Bootstrap CI failed: {e}")

    return final_pipe, cv_summary

def predict_with_regression(model, df_features):
    """
    Predict yields given a trained regression pipeline.

    Args:
        model: sklearn Pipeline (e.g., final_pipe returned from train_regression)
        df_features: pd.DataFrame with columns ['A','D','rho']

    Returns:
        numpy array of predicted yields (same order as df_features)
    """
    if not {'A','D','rho'}.issubset(set(df_features.columns)):
        raise ValueError("df_features must contain columns: A, D, rho")
    X = df_features[['A','D','rho']].values
    return model.predict(X)
