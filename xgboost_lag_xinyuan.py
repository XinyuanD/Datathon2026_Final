import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

# ---- Config ----
DATA_PATH = "cleaned_health_data_xinyuan.csv"

BASELINE_PRED_YEAR = 2024          # generate predicted-vs-actual for this year
FUTURE_PRED_YEAR = 2025            # generate predictions for this year

OUT_PREDVS_2024_PATH = f"pred_vs_actual_{BASELINE_PRED_YEAR}_XGBoostLag.csv"
OUT_PRED_2025_PATH = f"pred_{FUTURE_PRED_YEAR}_XGBoostLag.csv"

TARGET_COL = "ESTIMATE"
YEAR_COL = "TIME_PERIOD"


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag/trend features per series (defined by all columns except TIME_PERIOD and ESTIMATE)."""
    df = df.copy()
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[YEAR_COL, TARGET_COL])

    key_cols = [c for c in df.columns if c not in [YEAR_COL, TARGET_COL]]

    df = df.sort_values(key_cols + [YEAR_COL])
    g = df.groupby(key_cols, dropna=False)

    df["lag_1"] = g[TARGET_COL].shift(1)
    df["lag_2"] = g[TARGET_COL].shift(2)
    df["diff_1"] = df["lag_1"] - df["lag_2"]

    df["roll_mean_3"] = g[TARGET_COL].transform(lambda s: s.shift(1).rolling(3).mean())
    df["roll_std_3"] = g[TARGET_COL].transform(lambda s: s.shift(1).rolling(3).std())

    for col in ["lag_1", "lag_2", "diff_1", "roll_mean_3", "roll_std_3"]:
        df[col] = df[col].fillna(df[col].median())

    return df


def make_future_rows_from_history(df_raw: pd.DataFrame, pred_year: int) -> pd.DataFrame:
    """Create rows for pred_year with lag/trend features computed ONLY from historical targets."""
    df_raw = df_raw.copy()
    df_raw[YEAR_COL] = pd.to_numeric(df_raw[YEAR_COL], errors="coerce")
    df_raw[TARGET_COL] = pd.to_numeric(df_raw[TARGET_COL], errors="coerce")
    df_raw = df_raw.dropna(subset=[YEAR_COL, TARGET_COL])

    key_cols = [c for c in df_raw.columns if c not in [YEAR_COL, TARGET_COL]]
    df_raw = df_raw.sort_values(key_cols + [YEAR_COL])

    def feat_from_series(s: pd.Series) -> pd.Series:
        vals = s.values
        lag_1 = vals[-1] if len(vals) >= 1 else np.nan
        lag_2 = vals[-2] if len(vals) >= 2 else np.nan
        last3 = vals[-3:] if len(vals) >= 1 else np.array([np.nan])

        roll_mean_3 = float(np.nanmean(last3)) if len(last3) else np.nan
        roll_std_3 = float(np.nanstd(last3, ddof=1)) if len(last3) >= 2 else 0.0
        diff_1 = float(lag_1 - lag_2) if len(vals) >= 2 else 0.0

        return pd.Series({
            "lag_1": lag_1,
            "lag_2": lag_2 if not pd.isna(lag_2) else lag_1,
            "diff_1": diff_1,
            "roll_mean_3": roll_mean_3,
            "roll_std_3": roll_std_3,
        })

    feats_long = df_raw.groupby(key_cols, dropna=False)[TARGET_COL].apply(feat_from_series)
    feats_wide = feats_long.unstack(-1).reset_index()

    latest = (
        df_raw.groupby(key_cols, dropna=False)
        .tail(1)
        .drop(columns=[TARGET_COL])
        .copy()
    )
    latest[YEAR_COL] = pred_year

    pred_df = latest.merge(feats_wide, on=key_cols, how="left")

    for col in ["lag_1", "lag_2", "diff_1", "roll_mean_3", "roll_std_3"]:
        if col not in pred_df.columns:
            pred_df[col] = np.nan
        pred_df[col] = pred_df[col].fillna(pred_df[col].median())

    return pred_df


def build_model(X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    model = XGBRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=8,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe


def main():
    df = pd.read_csv(DATA_PATH)
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # ---- Train model on all observed years (with lag features) ----
    df_feat = add_lag_features(df)
    train_df = df_feat.dropna(subset=[TARGET_COL]).copy()

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].values

    pipe = build_model(X_train, y_train)

    # ---- (1) Predicted vs Actual for 2024 ----
    test_2024 = df_feat[df_feat[YEAR_COL] == BASELINE_PRED_YEAR].copy()
    X_2024 = test_2024.drop(columns=[TARGET_COL])
    y_2024 = test_2024[TARGET_COL].values

    pred_2024 = pipe.predict(X_2024)

    rmse = float(np.sqrt(mean_squared_error(y_2024, pred_2024)))
    mae = float(mean_absolute_error(y_2024, pred_2024))
    r2 = float(r2_score(y_2024, pred_2024))
    print(f"{BASELINE_PRED_YEAR} results (XGBoost + lag features): RMSE={rmse:.6f}  MAE={mae:.6f}  R2={r2:.6f}")

    out_2024 = test_2024.drop(columns=[TARGET_COL]).copy()
    out_2024["ACTUAL_ESTIMATE"] = y_2024
    out_2024["PRED_ESTIMATE"] = pred_2024
    out_2024["ERROR"] = out_2024["PRED_ESTIMATE"] - out_2024["ACTUAL_ESTIMATE"]
    out_2024["ABS_ERROR"] = np.abs(out_2024["ERROR"])
    out_2024.to_csv(OUT_PREDVS_2024_PATH, index=False)
    print("Wrote:", OUT_PREDVS_2024_PATH)

    # ---- (2) Predict 2025 ----
    pred_df_2025 = make_future_rows_from_history(df, FUTURE_PRED_YEAR)
    pred_2025 = pipe.predict(pred_df_2025)

    out_2025 = pred_df_2025.copy()
    out_2025["PRED_ESTIMATE"] = pred_2025
    out_2025.to_csv(OUT_PRED_2025_PATH, index=False)
    print("Wrote:", OUT_PRED_2025_PATH)


if __name__ == "__main__":
    main()
