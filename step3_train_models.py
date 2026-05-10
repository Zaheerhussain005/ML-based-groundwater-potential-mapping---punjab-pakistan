"""
Step 3: Train and evaluate ML models for groundwater potential classification.

Models trained:
  1. Random Forest        (bagging baseline, OOB validation)
  2. XGBoost              (gradient boosting, hist algorithm, early stopping)
  3. LightGBM             (gradient boosting, leaf-wise, early stopping)
  4. Soft-Voting Ensemble (probability average of the three)

Evaluation:
  - 80/20 stratified train-test split
  - 5-fold stratified cross-validation (macro F1)
  - Per-class precision / recall / F1
  - ROC-AUC (One-vs-Rest)  + ROC curves
  - Cohen's Kappa, Matthews Correlation Coefficient
  - Confusion matrices, Precision-Recall curves
  - Calibration / reliability diagrams
  - Feature importance (impurity / gain) for each model
  - Cross-model comparison bar charts and CV-fold boxplots

Outputs:
  outputs/models/
    random_forest.joblib
    xgboost.joblib
    lightgbm.joblib
    voting_ensemble.joblib
    best_model.joblib              (copy of top performer by macro F1)
    metrics.json                   (all numeric results)
    feature_importance.csv
    label_encoder_info.json        (column order + class names for prediction)
  outputs/figures/training/
    confusion_matrices.png
    roc_curves.png
    pr_curves.png
    feature_importance.png
    model_comparison.png
    cv_fold_scores.png
    calibration_curves.png
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef, f1_score,
    precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────
PROCESSED   = Path('data/processed')
MODELS_DIR  = Path('outputs/models')
FIG_DIR     = Path('outputs/figures/training')
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ['Elevation','Slope','Aspect','TWI','NDVI','NDWI','Rainfall','Soil','LST']
TARGET   = 'GW_Potential'
CLASS_NAMES   = ['Low', 'Medium', 'High']
CLASS_COLORS  = ['#d73027', '#fee090', '#1a9850']
RANDOM_STATE  = 42

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ════════════════════════════════════════════════════════════════════════════
#  Data loading + split
# ════════════════════════════════════════════════════════════════════════════
def load_data():
    path = PROCESSED / 'training_samples.csv'
    logger.info(f'Loading {path} ...')
    df = pd.read_csv(path)
    X = df[FEATURES].astype('float32').values
    y = df[TARGET].astype('int8').values
    logger.info(f'Shape: X={X.shape}  y={y.shape}  classes={dict(zip(*np.unique(y, return_counts=True)))}')
    return X, y


def split(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    # Carve a validation slice out of train for boosting early stopping.
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.10, stratify=y_tr, random_state=RANDOM_STATE
    )
    logger.info(f'Train={len(X_tr2):,}  Val={len(X_val):,}  Test={len(X_te):,}')
    return X_tr, X_te, y_tr, y_te, X_tr2, X_val, y_tr2, y_val


# ════════════════════════════════════════════════════════════════════════════
#  Model trainers
# ════════════════════════════════════════════════════════════════════════════
def train_rf(X_tr, y_tr):
    logger.info('Training Random Forest (300 trees, max_depth=22, OOB) ...')
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=22,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        oob_score=True,
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr, y_tr)
    dt = time.time() - t0
    logger.success(f'  RF trained in {dt:.1f}s | OOB score = {model.oob_score_:.4f}')
    return model


def train_xgb(X_tr, y_tr, X_val, y_val):
    logger.info('Training XGBoost (hist, early stopping, 800 max iters) ...')
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    dt = time.time() - t0
    logger.success(f'  XGB trained in {dt:.1f}s | best_iteration = {model.best_iteration}')
    return model


def train_lgbm(X_tr, y_tr, X_val, y_val):
    logger.info('Training LightGBM (gbdt, early stopping, 800 max iters) ...')
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=800,
        num_leaves=127,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        objective='multiclass',
        num_class=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    dt = time.time() - t0
    logger.success(f'  LGBM trained in {dt:.1f}s | best_iteration = {model.best_iteration_}')
    return model


def build_voting(rf, xgb_m, lgbm_m, X_tr, y_tr):
    logger.info('Building soft-voting ensemble (RF + XGB + LGBM) ...')
    t0 = time.time()
    # Use prefit estimators by wrapping their predict_proba into a custom object;
    # cleanest path with sklearn's VotingClassifier is to re-fit all three, but
    # they're already fit — so we'll build a thin wrapper that computes the
    # average probability without re-training. This wrapper is picklable.
    ensemble = SoftVoting([('rf', rf), ('xgb', xgb_m), ('lgbm', lgbm_m)])
    ensemble.fit(X_tr, y_tr)  # captures classes_, no real training
    dt = time.time() - t0
    logger.success(f'  Ensemble assembled in {dt:.2f}s')
    return ensemble


class SoftVoting:
    """Lightweight soft-voting wrapper around already-fitted classifiers.
    Picklable; saved by joblib alongside referenced models."""
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        probs = [est.predict_proba(X) for _, est in self.estimators]
        return np.mean(probs, axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ════════════════════════════════════════════════════════════════════════════
def evaluate(name, model, X_te, y_te):
    logger.info(f'Evaluating {name} ...')
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)

    metrics = {
        'accuracy'        : float(accuracy_score(y_te, y_pred)),
        'f1_macro'        : float(f1_score(y_te, y_pred, average='macro')),
        'f1_weighted'     : float(f1_score(y_te, y_pred, average='weighted')),
        'precision_macro' : float(precision_score(y_te, y_pred, average='macro')),
        'recall_macro'    : float(recall_score(y_te, y_pred, average='macro')),
        'cohen_kappa'     : float(cohen_kappa_score(y_te, y_pred)),
        'mcc'             : float(matthews_corrcoef(y_te, y_pred)),
        'roc_auc_ovr'     : float(roc_auc_score(y_te, y_proba, multi_class='ovr', average='macro')),
        'per_class': {},
    }

    p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred)
    for i, cn in enumerate(CLASS_NAMES):
        metrics['per_class'][cn] = {
            'precision': float(p[i]),
            'recall'   : float(r[i]),
            'f1'       : float(f1[i]),
        }

    metrics['confusion_matrix'] = confusion_matrix(y_te, y_pred).tolist()

    logger.info(f'  Accuracy:    {metrics["accuracy"]:.4f}')
    logger.info(f'  F1 (macro):  {metrics["f1_macro"]:.4f}')
    logger.info(f'  ROC-AUC OvR: {metrics["roc_auc_ovr"]:.4f}')
    logger.info(f'  Cohen K:     {metrics["cohen_kappa"]:.4f}')
    logger.info(f'  MCC:         {metrics["mcc"]:.4f}')
    print(classification_report(y_te, y_pred, target_names=CLASS_NAMES, digits=4))
    return metrics, y_pred, y_proba


def precision_recall_fscore_support(y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support as _prf
    return _prf(y_true, y_pred, labels=[0, 1, 2], zero_division=0)


def cv_scores(name, model_factory, X, y, n_splits=5):
    """Run StratifiedKFold CV; refits a fresh model per fold."""
    logger.info(f'5-fold CV — {name} ...')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fi, (tr, va) in enumerate(skf.split(X, y), 1):
        m = model_factory()
        m.fit(X[tr], y[tr])
        s = f1_score(y[va], m.predict(X[va]), average='macro')
        scores.append(s)
        logger.info(f'  fold {fi}: F1 macro = {s:.4f}')
    scores = np.array(scores)
    logger.info(f'  Mean F1 = {scores.mean():.4f}  Std = {scores.std():.4f}')
    return scores


# ════════════════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(results, y_te):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1: axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        cm = np.array(r['metrics']['confusion_matrix'])
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([[f'{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)'
                            for j in range(3)] for i in range(3)])
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, cbar=False, linewidths=0.5, linecolor='white')
        ax.set_title(f'{name}\nAccuracy = {r["metrics"]["accuracy"]:.4f}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.suptitle(f'Confusion Matrices — Test Set ({len(y_te):,} samples)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'confusion_matrices.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    logger.success('  confusion_matrices.png')


def plot_roc_curves(results, y_te):
    y_bin = label_binarize(y_te, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1: axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        for i, (cn, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], r['proba'][:, i])
            auc = roc_auc_score(y_bin[:, i], r['proba'][:, i])
            ax.plot(fpr, tpr, color=color, linewidth=2.2,
                    label=f'{cn} (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name}\nMacro AUC = {r["metrics"]["roc_auc_ovr"]:.4f}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle('ROC Curves — One-vs-Rest', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'roc_curves.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    logger.success('  roc_curves.png')


def plot_pr_curves(results, y_te):
    y_bin = label_binarize(y_te, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1: axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        for i, (cn, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], r['proba'][:, i])
            ap = average_precision_score(y_bin[:, i], r['proba'][:, i])
            ax.plot(rec, prec, color=color, linewidth=2.2,
                    label=f'{cn} (AP = {ap:.4f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.02)
    plt.suptitle('Precision–Recall Curves', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pr_curves.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    logger.success('  pr_curves.png')


def plot_feature_importance(rf, xgb_m, lgbm_m):
    fi = pd.DataFrame({
        'Feature' : FEATURES,
        'RandomForest' : rf.feature_importances_,
        'XGBoost'      : xgb_m.feature_importances_,
        'LightGBM'     : lgbm_m.feature_importances_ / lgbm_m.feature_importances_.sum(),
    })
    fi['Mean'] = fi[['RandomForest', 'XGBoost', 'LightGBM']].mean(axis=1)
    fi = fi.sort_values('Mean', ascending=True)
    fi.to_csv(MODELS_DIR / 'feature_importance.csv', index=False)
    logger.info('  feature_importance.csv saved')

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=True)
    for ax, col, color in zip(axes, ['RandomForest', 'XGBoost', 'LightGBM', 'Mean'],
                              ['#1f77b4', '#d62728', '#2ca02c', '#444444']):
        ax.barh(fi['Feature'], fi[col], color=color, edgecolor='black', linewidth=0.6)
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel('Relative Importance')
        ax.grid(True, axis='x', alpha=0.3)
        for i, v in enumerate(fi[col]):
            ax.text(v + max(fi[col]) * 0.01, i, f'{v:.3f}',
                    va='center', fontsize=8)
    plt.suptitle('Feature Importance — Cross-Model Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'feature_importance.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    logger.success('  feature_importance.png')
    return fi


def plot_model_comparison(results):
    keys = ['accuracy', 'f1_macro', 'roc_auc_ovr', 'cohen_kappa', 'mcc']
    titles = ['Accuracy', 'F1 (macro)', 'ROC-AUC OvR', "Cohen's Kappa", 'MCC']
    df = pd.DataFrame({n: [r['metrics'][k] for k in keys] for n, r in results.items()},
                      index=titles)

    fig, ax = plt.subplots(figsize=(11, 6))
    df.plot(kind='bar', ax=ax, edgecolor='black', linewidth=0.7, width=0.8)
    ax.set_title('Model Performance Comparison — Test Set',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Score')
    ax.set_xticklabels(titles, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Model', loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f', fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'model_comparison.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    logger.success('  model_comparison.png')


def plot_cv_box(cv_results):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data   = list(cv_results.values())
    labels = list(cv_results.keys())
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                    medianprops={'color': 'black', 'linewidth': 1.6})
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#d62728', '#2ca02c']):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for i, scores in enumerate(data, 1):
        ax.scatter([i] * len(scores), scores, color='black', zorder=3, s=22)
        ax.text(i, max(scores) + 0.002,
                f'mean = {np.mean(scores):.4f}\nstd = {np.std(scores):.4f}',
                ha='center', fontsize=9)
    ax.set_ylabel('Macro F1 score')
    ax.set_title('5-Fold Cross-Validation — Macro F1 Distribution',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'cv_fold_scores.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    logger.success('  cv_fold_scores.png')


def plot_calibration(results, y_te):
    """Reliability diagram for class 2 (High) — most decision-relevant class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect calibration')
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    y_bin_high = (y_te == 2).astype(int)
    for (name, r), color in zip(results.items(), colors):
        prob_high = r['proba'][:, 2]
        frac_pos, mean_pred = calibration_curve(y_bin_high, prob_high, n_bins=15)
        ax.plot(mean_pred, frac_pos, marker='o', linewidth=2,
                color=color, label=name)
    ax.set_xlabel('Mean predicted probability (High class)')
    ax.set_ylabel('Fraction of actual positives')
    ax.set_title('Calibration / Reliability Diagram — High GW Potential',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'calibration_curves.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    logger.success('  calibration_curves.png')


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 70)
    print('  Step 3: ML Model Training — Groundwater Potential Mapping')
    print('=' * 70)

    X, y = load_data()
    X_tr, X_te, y_tr, y_te, X_tr2, X_val, y_tr2, y_val = split(X, y)

    # ── Train base models ──────────────────────────────────────────────────
    rf      = train_rf(X_tr2, y_tr2)
    xgb_m   = train_xgb(X_tr2, y_tr2, X_val, y_val)
    lgbm_m  = train_lgbm(X_tr2, y_tr2, X_val, y_val)
    voting  = build_voting(rf, xgb_m, lgbm_m, X_tr2, y_tr2)

    # ── Evaluate on held-out test set ──────────────────────────────────────
    results = {}
    for name, model in [
        ('Random Forest', rf),
        ('XGBoost',       xgb_m),
        ('LightGBM',      lgbm_m),
        ('Voting',        voting),
    ]:
        m, p, pr = evaluate(name, model, X_te, y_te)
        results[name] = {'metrics': m, 'pred': p, 'proba': pr}

    # ── 5-fold CV (re-fit each base model fresh per fold; fast) ───────────
    cv_results = {
        'Random Forest': cv_scores(
            'RF',
            lambda: RandomForestClassifier(
                n_estimators=200, max_depth=20, max_features='sqrt',
                min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE,
                class_weight='balanced'),
            X, y),
        'XGBoost': cv_scores(
            'XGB',
            lambda: xgb.XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, tree_method='hist',
                objective='multi:softprob', num_class=3, n_jobs=-1,
                random_state=RANDOM_STATE, eval_metric='mlogloss'),
            X, y),
        'LightGBM': cv_scores(
            'LGBM',
            lambda: lgb.LGBMClassifier(
                n_estimators=300, num_leaves=127, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, n_jobs=-1,
                random_state=RANDOM_STATE, class_weight='balanced',
                verbosity=-1),
            X, y),
    }

    # ── Plots ──────────────────────────────────────────────────────────────
    logger.info('Generating plots ...')
    plot_confusion_matrices(results, y_te)
    plot_roc_curves(results, y_te)
    plot_pr_curves(results, y_te)
    fi_df = plot_feature_importance(rf, xgb_m, lgbm_m)
    plot_model_comparison(results)
    plot_cv_box(cv_results)
    plot_calibration(results, y_te)

    # ── Save models ────────────────────────────────────────────────────────
    logger.info('Saving models ...')
    joblib.dump(rf,      MODELS_DIR / 'random_forest.joblib',  compress=3)
    joblib.dump(xgb_m,   MODELS_DIR / 'xgboost.joblib',        compress=3)
    joblib.dump(lgbm_m,  MODELS_DIR / 'lightgbm.joblib',       compress=3)
    joblib.dump(voting,  MODELS_DIR / 'voting_ensemble.joblib', compress=3)

    # Pick best by macro F1
    name_map = {'Random Forest': 'random_forest', 'XGBoost': 'xgboost',
                'LightGBM': 'lightgbm', 'Voting': 'voting_ensemble'}
    best_name = max(results, key=lambda n: results[n]['metrics']['f1_macro'])
    best_path = MODELS_DIR / f'{name_map[best_name]}.joblib'
    joblib.dump(joblib.load(best_path), MODELS_DIR / 'best_model.joblib', compress=3)
    logger.success(f'Best model: {best_name} (macro F1 = {results[best_name]["metrics"]["f1_macro"]:.4f})')

    # ── Save metrics JSON + label encoder info ─────────────────────────────
    out = {
        'config': {
            'features'    : FEATURES,
            'class_names' : CLASS_NAMES,
            'random_state': RANDOM_STATE,
            'n_train'     : int(len(X_tr2)),
            'n_val'       : int(len(X_val)),
            'n_test'      : int(len(X_te)),
        },
        'rf_oob_score': float(rf.oob_score_),
        'xgb_best_iteration' : int(xgb_m.best_iteration),
        'lgbm_best_iteration': int(lgbm_m.best_iteration_),
        'test_metrics' : {n: r['metrics'] for n, r in results.items()},
        'cv_macro_f1'  : {n: {'mean': float(np.mean(s)), 'std': float(np.std(s)),
                              'folds': [float(x) for x in s]}
                          for n, s in cv_results.items()},
        'best_model': best_name,
    }
    with open(MODELS_DIR / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2)
    logger.success(f'  metrics.json')

    with open(MODELS_DIR / 'label_encoder_info.json', 'w') as f:
        json.dump({'features': FEATURES, 'class_names': CLASS_NAMES}, f, indent=2)
    logger.success(f'  label_encoder_info.json')

    # ── Final summary ──────────────────────────────────────────────────────
    print()
    print('=' * 70)
    print('  TRAINING COMPLETE')
    print('=' * 70)
    print(f'  Train / Val / Test : {len(X_tr2):,} / {len(X_val):,} / {len(X_te):,}')
    print()
    print('  Test-set performance (macro):')
    print(f'  {"Model":<16} {"Acc":>8} {"F1":>8} {"AUC":>8} {"Kappa":>8} {"MCC":>8}')
    for n, r in results.items():
        m = r['metrics']
        print(f'  {n:<16} {m["accuracy"]:>8.4f} {m["f1_macro"]:>8.4f} '
              f'{m["roc_auc_ovr"]:>8.4f} {m["cohen_kappa"]:>8.4f} {m["mcc"]:>8.4f}')
    print()
    print(f'  Best model         : {best_name}')
    print(f'  Models             : outputs/models/')
    print(f'  Plots              : outputs/figures/training/')
    print(f'  Metrics            : outputs/models/metrics.json')
    print()
    print('  Top features (mean importance across 3 models):')
    for _, row in fi_df.sort_values('Mean', ascending=False).iterrows():
        print(f'    {row["Feature"]:<10} {row["Mean"]:.4f}')


if __name__ == '__main__':
    main()
