import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.linear_model import LassoCV, Lasso
from imblearn.over_sampling import SMOTE


def test_notebook_preprocessing_pipeline_no_leakage_and_aligned_features():
    # === Exact dataset + split settings from your notebook ===
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()

    df, df_t = train_test_split(df, test_size=0.15, stratify=df["target"], random_state=42)

    # Define x/y and x_t/y_t exactly like your notebook
    x = df.drop("target", axis=1)
    y = df["target"]
    x_t = df_t.drop("target", axis=1)
    y_t = df_t["target"]

    # === SMOTE (train only) — same hyperparams ===
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    x_train_smote, y_train_smote = smote.fit_resample(x, y)
    x = x_train_smote
    y = y_train_smote

    # === VarianceThreshold — same threshold ===
    X = x
    Y = y

    selector0 = VarianceThreshold(threshold=0.01).fit(X, Y)
    selected_features_mask = selector0.get_support()
    selected_features = X.columns[selected_features_mask]
    X = pd.DataFrame(selector0.transform(X), columns=selected_features)

    # Keep df/df_t synced (your “if clause” idea, but applied directly here)
    low_var_cols = [c for c in x.columns if c not in selected_features]
    df.drop(columns=[c for c in low_var_cols if c in df.columns], inplace=True, errors="ignore")
    df_t.drop(columns=[c for c in low_var_cols if c in df_t.columns], inplace=True, errors="ignore")

    # === Correlated feature removal — same logic + threshold (0.8) ===
    y_encoded = Y
    corr_matrix = X.corr(method="pearson")

    auc_dict = {}
    for feature in X.columns:
        try:
            auc = roc_auc_score(y_encoded, X[feature])
            if auc < 0.5:
                auc = 1 - auc
            auc_dict[feature] = auc
        except Exception:
            auc_dict[feature] = 0.5

    threshold = 0.8
    features = X.columns.tolist()
    features_to_remove = set()

    for i in range(len(features)):
        for j in range(i):
            if features[i] in features_to_remove or features[j] in features_to_remove:
                continue
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                auc_i = auc_dict[features[i]]
                auc_j = auc_dict[features[j]]
                if auc_i < auc_j:
                    features_to_remove.add(features[i])
                else:
                    features_to_remove.add(features[j])

    cols_to_drop = list(features_to_remove)
    x_reduced = X.drop(columns=cols_to_drop, errors="ignore")
    x_t_reduced = x_t.drop(columns=cols_to_drop, errors="ignore")

    # Same “train/test column alignment” safety as your notebook
    missing_in_test = set(x_reduced.columns) - set(x_t_reduced.columns)
    for c in missing_in_test:
        x_t_reduced[c] = 0.0

    extra_in_test = set(x_t_reduced.columns) - set(x_reduced.columns)
    if extra_in_test:
        x_t_reduced = x_t_reduced.drop(columns=list(extra_in_test))

    x_t_reduced = x_t_reduced[x_reduced.columns]

    # Keep df/df_t synced (same as your cells 19/20)
    df.drop(columns=list(features_to_remove), axis=1, inplace=True)
    df_t.drop(columns=list(features_to_remove), axis=1, inplace=True)

    X = x_reduced.copy()
    X_t = x_t_reduced.copy()

    # Notebook safety
    X.columns = X.columns.astype(str)

    # === LASSO selection — same scaling + CV settings ===
    cols = X.columns
    scaler1 = StandardScaler()
    scaler2 = RobustScaler()
    scaler3 = QuantileTransformer()

    X_scaled_array = scaler1.fit_transform(X)
    X_test_scaled_array = scaler1.transform(X_t)

    X = pd.DataFrame(X_scaled_array, index=X.index, columns=cols)
    X_t = pd.DataFrame(X_test_scaled_array, index=X_t.index, columns=cols)

    n_splits = 15
    cv = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

    candidate_alphas = np.logspace(-6, 0, 100)

    lasso_cv = LassoCV(
        alphas=candidate_alphas,
        cv=cv,
        max_iter=10000,
        tol=0.001,
        random_state=42,
    )
    lasso_cv.fit(X, Y)

    mse_path = lasso_cv.mse_path_
    mean_mse = np.mean(mse_path, axis=1)
    optimal_idx = np.argmin(mean_mse)
    best_alpha = lasso_cv.alphas_[optimal_idx]

    lasso = Lasso(alpha=best_alpha, random_state=42).fit(X, Y)
    model = SelectFromModel(lasso, prefit=True)

    X_lasso_selected = model.transform(X)
    mask = model.get_support()
    selected_features = X.columns[mask]

    df_selected_features = pd.DataFrame(X_lasso_selected, index=X.index, columns=selected_features)

    X = df_selected_features
    X_t = X_t[selected_features]  # ensure test set has same selected features

    # === RobustScaler before model training — exactly like your cell 28 ===
    scaler2 = RobustScaler()
    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

    # Critical contract: same columns, same order
    assert list(X.columns) == list(X_t.columns)

    X_train = X.values
    X_train_scaled = scaler2.fit_transform(X_train)

    X_test = X_t.values
    X_test_scaled = scaler2.transform(X_test)

    # === What this integration test checks ===

    # 1) No leakage: RobustScaler fit on TRAIN only
    train_medians = np.median(X_train, axis=0)
    np.testing.assert_allclose(scaler2.center_, train_medians, rtol=0, atol=1e-10)

    # 2) No leakage: SMOTE is train-only, so test row count stays the same
    assert X_test_scaled.shape[0] == len(y_t)

    # 3) Shapes are consistent + outputs are finite
    assert X_train_scaled.shape[1] == X_test_scaled.shape[1] == X.shape[1] > 0
    assert np.isfinite(X_train_scaled).all()
    assert np.isfinite(X_test_scaled).all()
    