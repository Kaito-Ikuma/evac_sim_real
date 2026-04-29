import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_log10(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log10(np.clip(np.asarray(x, dtype=float), eps, None))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def rankdata_average(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    rx = rankdata_average(x[m])
    ry = rankdata_average(y[m])
    return float(np.corrcoef(rx, ry)[0, 1])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return np.nan
    yt = y_true[m]
    yp = y_pred[m]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def adjusted_r2(r2: float, n: int, p: int) -> float:
    if not np.isfinite(r2) or n <= p + 1:
        return np.nan
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def fit_ols(X: np.ndarray, y: np.ndarray):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    n, p = X.shape
    rss = np.sum((y - yhat) ** 2)
    sigma2 = rss / max(n - p, 1)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(sigma2 * xtx_inv), 0, None))
    return beta, se, yhat


def standardize_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        mu = df[c].mean()
        sd = df[c].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            out[c] = 0.0
        else:
            out[c] = (df[c] - mu) / sd
    return out


def pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Required columns not found. Tried: {candidates}")
    return None


def make_scatter(y_true, y_pred, outpath: Path, title: str, xlabel: str, loglog: bool = False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_pred, y_true, alpha=0.45, s=20)
    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.any():
        lo = float(np.nanmin(np.concatenate([x[m], y[m]])))
        hi = float(np.nanmax(np.concatenate([x[m], y[m]])))
        if loglog:
            lo = max(lo, 1e-6)
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Observed $\tau_i^{move}$")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def make_component_panel(df: pd.DataFrame, tau_col: str, outpath: Path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    specs = [
        ("V_at_decision", "Distance vs tau_move", "V_at_decision"),
        ("post_no_progress_frames", "No-progress vs tau_move", "post_no_progress_frames"),
        ("post_stagnated_frames", "Stagnated vs tau_move", "post_stagnated_frames"),
        ("lateral_no_progress", "Lateral no-progress vs tau_move", "lateral_no_progress"),
    ]
    for ax, (col, title, xlabel) in zip(axes, specs):
        ax.scatter(df[col], df[tau_col], alpha=0.45, s=16)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(tau_col)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def make_pred_comparison(df: pd.DataFrame, tau_col: str, pred_cols: list[str], outpath: Path):
    fig, axes = plt.subplots(1, len(pred_cols), figsize=(7 * len(pred_cols), 6))
    if len(pred_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, pred_cols):
        ax.scatter(df[col], df[tau_col], alpha=0.4, s=16)
        x = df[col].to_numpy(float)
        y = df[tau_col].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.any():
            lo = float(np.nanmin(np.concatenate([x[m], y[m]])))
            hi = float(np.nanmax(np.concatenate([x[m], y[m]])))
            ax.plot([lo, hi], [lo, hi], linestyle='--')
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel(tau_col)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Validate tau_move ~ a1 V_dec + a2 N_no_progress + a3 N_stagnated using tau_move_validation_table_postmean.csv")
    parser.add_argument("--csv", type=str, default="tau_move_validation_table_postmean.csv")
    parser.add_argument("--output_dir", type=str, default="tau_move_linear_decomp_results")
    parser.add_argument("--force_nonnegative", action="store_true", help="Fit nonnegative coefficients by clipping negative OLS coefficients to zero and refitting intercept.")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    tau_col = pick_column(df, ["tau_move", "observed_tau_move", "tau_move_obs"])
    v_col = pick_column(df, ["V_at_decision", "V_dec", "V_at_decision_postmean"])
    no_col = pick_column(df, ["post_no_progress_frames", "no_progress_post", "post_no_progress"])
    stag_col = pick_column(df, ["post_stagnated_frames", "stagnated_post", "post_stagnated"])

    # Optional existing columns useful for comparison
    raw_old_col = pick_column(df, ["tau_pred_fit_l2", "tau_pred_fit_geom", "tau_pred_raw"], required=False)

    work = df.copy()
    work["lateral_no_progress"] = np.clip(work[no_col] - work[stag_col], 0, None)

    required_cols = [tau_col, v_col, no_col, stag_col, "lateral_no_progress"]
    mask = np.ones(len(work), dtype=bool)
    for c in required_cols:
        mask &= np.isfinite(work[c].to_numpy(dtype=float))
    mask &= work[tau_col].to_numpy(dtype=float) >= 0
    valid = work.loc[mask].copy()

    if len(valid) < 10:
        raise ValueError("Not enough valid rows for regression.")

    y = valid[tau_col].to_numpy(dtype=float)

    model_specs = {
        "M1_V_only": [v_col],
        "M2_V_plus_no": [v_col, no_col],
        "M3_V_plus_stag": [v_col, stag_col],
        "M4_V_plus_no_plus_stag": [v_col, no_col, stag_col],
        "M5_V_plus_lateral_plus_stag": [v_col, "lateral_no_progress", stag_col],
    }

    summary_rows = []
    pred_table = valid[[tau_col, v_col, no_col, stag_col, "lateral_no_progress"]].copy()

    for model_name, feat_cols in model_specs.items():
        X_df = valid[feat_cols].copy()
        X = np.column_stack([np.ones(len(X_df)), X_df.to_numpy(dtype=float)])
        beta, se, yhat = fit_ols(X, y)

        if args.force_nonnegative:
            beta_nn = beta.copy()
            beta_nn[1:] = np.clip(beta_nn[1:], 0, None)
            # Refit intercept with slopes fixed
            intercept = float(np.mean(y - X_df.to_numpy(dtype=float) @ beta_nn[1:]))
            beta_nn[0] = intercept
            beta = beta_nn
            yhat = intercept + X_df.to_numpy(dtype=float) @ beta[1:]
            se = np.full_like(beta, np.nan, dtype=float)

        r2 = r2_score(y, yhat)
        ar2 = adjusted_r2(r2, len(y), X.shape[1] - 1)
        pear = safe_corr(y, yhat)
        spear = spearman_corr(y, yhat)
        pear_log = safe_corr(safe_log10(y), safe_log10(yhat))
        spear_log = spearman_corr(safe_log10(y), safe_log10(yhat))

        row = {
            "model": model_name,
            "features": ", ".join(feat_cols),
            "n": len(y),
            "r2": r2,
            "adjusted_r2": ar2,
            "pearson": pear,
            "spearman": spear,
            "pearson_log": pear_log,
            "spearman_log": spear_log,
            "intercept": beta[0],
        }
        for j, c in enumerate(feat_cols, start=1):
            row[f"coef_{c}"] = beta[j]
            row[f"se_{c}"] = se[j] if j < len(se) else np.nan
        summary_rows.append(row)

        pred_col = f"pred_{model_name}"
        pred_table[pred_col] = yhat

        make_scatter(
            y, yhat,
            outdir / f"{pred_col}_linear.png",
            title=f"tau_move validation: {model_name}",
            xlabel=f"Predicted {model_name}",
            loglog=False,
        )
        make_scatter(
            y, yhat,
            outdir / f"{pred_col}_loglog.png",
            title=f"tau_move validation (log-log): {model_name}",
            xlabel=f"Predicted {model_name}",
            loglog=True,
        )

    # Standardized-coefficient fit for the recommended model M5
    zcols = [v_col, "lateral_no_progress", stag_col]
    Z = standardize_cols(valid, zcols)
    Zy = (valid[tau_col] - valid[tau_col].mean()) / (valid[tau_col].std(ddof=0) if valid[tau_col].std(ddof=0) != 0 else 1.0)
    Xz = np.column_stack([np.ones(len(Z)), Z.to_numpy(dtype=float)])
    beta_z, se_z, yhat_z = fit_ols(Xz, Zy.to_numpy(dtype=float))
    std_row = {
        "model": "M5_standardized",
        "features": ", ".join(zcols),
        "n": len(Z),
        "intercept": beta_z[0],
        **{f"beta_std_{c}": beta_z[i+1] for i, c in enumerate(zcols)},
        **{f"se_std_{c}": se_z[i+1] for i, c in enumerate(zcols)},
    }

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "tau_move_linear_models_summary.csv", index=False)
    pd.DataFrame([std_row]).to_csv(outdir / "tau_move_standardized_coefficients.csv", index=False)
    pred_table.to_csv(outdir / "tau_move_linear_decomp_predictions.csv", index=False)

    make_component_panel(valid, tau_col, outdir / "tau_move_linear_components_panel.png")
    comparison_cols = [c for c in [raw_old_col, "pred_M4_V_plus_no_plus_stag", "pred_M5_V_plus_lateral_plus_stag"] if c is not None and c in pred_table.columns]
    if comparison_cols:
        tmp = pred_table.join(valid[[tau_col]], how="left", rsuffix="_obs").copy()
        tmp[tau_col] = valid[tau_col].to_numpy()
        make_pred_comparison(tmp, tau_col, comparison_cols, outdir / "tau_move_prediction_comparison.png")

    # Human-readable report
    best_idx = summary_df["adjusted_r2"].astype(float).idxmax()
    best = summary_df.loc[best_idx]

    lines = []
    lines.append("=== tau_move linear decomposition validation ===")
    lines.append(f"input_csv: {args.csv}")
    lines.append(f"n_valid: {len(valid)}")
    lines.append("")
    lines.append("Variables used:")
    lines.append(f"  tau_move column          : {tau_col}")
    lines.append(f"  V_at_decision column     : {v_col}")
    lines.append(f"  post_no_progress column  : {no_col}")
    lines.append(f"  post_stagnated column    : {stag_col}")
    lines.append("  derived column           : lateral_no_progress = no_progress - stagnated (clipped at 0)")
    lines.append("")
    lines.append("Single-variable correlations with observed tau_move:")
    lines.append(f"  corr(V_at_decision, tau_move)                = {safe_corr(valid[v_col], valid[tau_col]):.6f}")
    lines.append(f"  corr(post_no_progress_frames, tau_move)      = {safe_corr(valid[no_col], valid[tau_col]):.6f}")
    lines.append(f"  corr(post_stagnated_frames, tau_move)        = {safe_corr(valid[stag_col], valid[tau_col]):.6f}")
    lines.append(f"  corr(lateral_no_progress, tau_move)          = {safe_corr(valid['lateral_no_progress'], valid[tau_col]):.6f}")
    lines.append(f"  corr(no_progress, stagnated)                 = {safe_corr(valid[no_col], valid[stag_col]):.6f}")
    lines.append("")
    lines.append("Model summary:")
    for _, r in summary_df.iterrows():
        lines.append(
            f"  {r['model']}: adjusted_R2={r['adjusted_r2']:.6f}, R2={r['r2']:.6f}, "
            f"pearson={r['pearson']:.6f}, spearman={r['spearman']:.6f}, "
            f"pearson_log={r['pearson_log']:.6f}, spearman_log={r['spearman_log']:.6f}"
        )
        feat_cols = [x.strip() for x in str(r['features']).split(',') if x.strip()]
        coeffs = [f"intercept={r['intercept']:.6f}"]
        for c in feat_cols:
            coeffs.append(f"{c}={r.get('coef_' + c, np.nan):.6f}")
        lines.append("     coefficients: " + ", ".join(coeffs))
    lines.append("")
    lines.append("Best model by adjusted R2:")
    lines.append(f"  {best['model']} with adjusted_R2={best['adjusted_r2']:.6f}")
    lines.append(f"  features: {best['features']}")
    lines.append("")
    lines.append("Standardized coefficients for M5 (V + lateral_no_progress + stagnated):")
    lines.append(f"  beta_std_{v_col} = {std_row[f'beta_std_{v_col}']:.6f}")
    lines.append(f"  beta_std_lateral_no_progress = {std_row['beta_std_lateral_no_progress']:.6f}")
    lines.append(f"  beta_std_{stag_col} = {std_row[f'beta_std_{stag_col}']:.6f}")
    lines.append("")
    lines.append("Saved files:")
    for p in sorted(outdir.glob("*")):
        lines.append(f"  - {p.name}")

    report_path = outdir / "analysis_summary_linear_decomp.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
