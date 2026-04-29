#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze tau_move using collective bottleneck models.

This script validates two models using existing outputs:

Model D (decision-frame merge):
    tau_move ~ V_at_decision + Q(t_dec) + rho_up(t_dec) + 1/(J_out(t_dec)+eps)

Model A (estimated-arrival merge):
    t_arr_est = t_dec + a1 * V_at_decision
    tau_move ~ V_at_decision + Q(t_arr_est) / (J_out(t_arr_est)+eps)

Inputs expected:
  - tau_move_validation_table_postmean.csv
      Must contain at least: tau_move, V_at_decision
      Ideally contains one of: decision_frame, step_when_decided, decision_h
  - macro_stats.csv
      Must contain: h_ext and/or frame, Q_transport or q_transport,
      rho_up if available, J_out_frame or J_out_per_step if available.

The script is intentionally robust to slightly different column names.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of these columns were found: {list(candidates)}\nAvailable columns: {list(df.columns)}")
    return None


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    tmp = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 3:
        return np.nan
    if tmp.iloc[:, 0].nunique() < 2 or tmp.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method=method))


def fit_ols(X: np.ndarray, y: np.ndarray, nonnegative: bool = False):
    """OLS with intercept. Optional nonnegative slopes by clipping after lstsq (simple diagnostic only)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    if nonnegative:
        # Simple projected diagnostic: keep intercept, clip slopes, refit intercept.
        beta[1:] = np.maximum(beta[1:], 0.0)
        intercept = np.mean(y - X @ beta[1:])
        beta[0] = intercept

    y_pred = X_design @ beta
    return beta, y_pred


def regression_metrics(y: np.ndarray, y_pred: np.ndarray, n_features: int) -> dict:
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(y_pred)
    y = y[mask]
    y_pred = y_pred[mask]
    n = len(y)
    if n < 3:
        return {"n": n, "R2": np.nan, "adj_R2": np.nan, "rmse": np.nan,
                "mae": np.nan, "pearson": np.nan, "spearman": np.nan,
                "pearson_log": np.nan, "spearman_log": np.nan}

    resid = y - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - n_features - 1, 1) if np.isfinite(r2) else np.nan
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    p = safe_corr(pd.Series(y_pred), pd.Series(y), "pearson")
    s = safe_corr(pd.Series(y_pred), pd.Series(y), "spearman")

    pos = (y > 0) & (y_pred > 0)
    if np.sum(pos) >= 3:
        plog = safe_corr(pd.Series(np.log(y_pred[pos])), pd.Series(np.log(y[pos])), "pearson")
        slog = safe_corr(pd.Series(np.log(y_pred[pos])), pd.Series(np.log(y[pos])), "spearman")
    else:
        plog = np.nan
        slog = np.nan

    return {"n": n, "R2": r2, "adj_R2": adj_r2, "rmse": rmse, "mae": mae,
            "pearson": p, "spearman": s, "pearson_log": plog, "spearman_log": slog}


def plot_pred(y: np.ndarray, y_pred: np.ndarray, title: str, xlabel: str, out_linear: str, out_log: str) -> None:
    mask = np.isfinite(y) & np.isfinite(y_pred)
    yy = y[mask]
    pp = y_pred[mask]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pp, yy, alpha=0.5, s=15)
    mn = np.nanmin([np.nanmin(pp), np.nanmin(yy), 0])
    mx = np.nanmax([np.nanmax(pp), np.nanmax(yy)])
    ax.plot([mn, mx], [mn, mx], linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Observed $\tau_i^{move}$")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_linear, dpi=200)
    plt.close(fig)

    pos = mask & (y > 0) & (y_pred > 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_pred[pos], y[pos], alpha=0.5, s=15)
    if np.sum(pos) > 0:
        mn = np.nanmin([np.nanmin(y_pred[pos]), np.nanmin(y[pos])])
        mx = np.nanmax([np.nanmax(y_pred[pos]), np.nanmax(y[pos])])
        ax.plot([mn, mx], [mn, mx], linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Observed $\tau_i^{move}$")
    ax.set_title(title + " (log-log)")
    fig.tight_layout()
    fig.savefig(out_log, dpi=200)
    plt.close(fig)


def smooth_series(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values.astype(float)
    return values.astype(float).rolling(window=window, center=True, min_periods=1).mean()


# ============================================================
# Loading and preparation
# ============================================================

@dataclass
class ColumnMap:
    tau: str
    V: str
    decision_frame: Optional[str]
    decision_h: Optional[str]
    step_when_decided: Optional[str]


def detect_agent_columns(agent_df: pd.DataFrame) -> ColumnMap:
    tau_col = first_existing_column(agent_df, ["tau_move", "tau_move_steps", "movement_delay"])
    V_col = first_existing_column(agent_df, ["V_at_decision", "V_dec", "V_i_dec"])
    dec_frame_col = first_existing_column(agent_df, ["decision_frame", "frame_decision", "dec_frame"], required=False)
    dec_h_col = first_existing_column(agent_df, ["decision_h", "h_at_decision", "h_decision"], required=False)
    step_dec_col = first_existing_column(agent_df, ["step_when_decided", "decision_step", "tau_dec_steps"], required=False)
    return ColumnMap(tau_col, V_col, dec_frame_col, dec_h_col, step_dec_col)


def prepare_macro(macro_df: pd.DataFrame, smooth_window: int, eps: Optional[float]) -> pd.DataFrame:
    out = macro_df.copy()

    if "frame" not in out.columns:
        out["frame"] = np.arange(len(out), dtype=int)

    if "h_ext" not in out.columns:
        raise ValueError("macro_stats.csv needs h_ext column for robust merging.")

    Q_col = first_existing_column(out, ["Q_transport", "Q", "queue", "transport_inventory", "q_transport"], required=False)
    if Q_col is None:
        if "m_dec_cum" in out.columns and "m_evac_cum" in out.columns:
            out["Q_transport"] = out["m_dec_cum"] - out["m_evac_cum"]
            Q_col = "Q_transport"
        elif "m_dec_cum" in out.columns and "m_evac" in out.columns:
            out["Q_transport"] = out["m_dec_cum"] - out["m_evac"]
            Q_col = "Q_transport"
        else:
            raise ValueError("Could not find or construct Q_transport in macro_stats.csv")
    out["Q_model"] = out[Q_col].astype(float)

    rho_col = first_existing_column(out, ["rho_up", "rho_upstream", "upstream_density"], required=False)
    if rho_col is not None:
        out["rho_up_model"] = out[rho_col].astype(float)
    else:
        out["rho_up_model"] = np.nan

    J_col = first_existing_column(
        out,
        ["J_out_frame", "J_out_per_step", "service_rate_event_frame", "exit_count_frame", "J_out"],
        required=False,
    )
    if J_col is None:
        # Construct from evacuation count if possible.
        if "m_evac_cum" in out.columns:
            out["J_out_frame"] = out["m_evac_cum"].diff().fillna(out["m_evac_cum"]).clip(lower=0)
            J_col = "J_out_frame"
        elif "m_evac" in out.columns:
            out["J_out_frame"] = out["m_evac"].diff().fillna(out["m_evac"]).clip(lower=0)
            J_col = "J_out_frame"
        else:
            raise ValueError("Could not find or construct J_out in macro_stats.csv")

    out["J_out_model"] = out[J_col].astype(float)
    out["J_out_smooth"] = smooth_series(out["J_out_model"], smooth_window)

    if eps is None:
        positive = out.loc[out["J_out_smooth"] > 0, "J_out_smooth"]
        if len(positive) > 0:
            eps_val = max(1e-9, 0.05 * float(positive.median()))
        else:
            eps_val = 1.0
    else:
        eps_val = float(eps)
    out.attrs["eps_J"] = eps_val
    out["inv_J_out"] = 1.0 / (out["J_out_smooth"] + eps_val)

    return out[["frame", "h_ext", "Q_model", "rho_up_model", "J_out_model", "J_out_smooth", "inv_J_out"]]


def add_decision_frame_if_needed(agent_df: pd.DataFrame, cmap: ColumnMap, macro: pd.DataFrame, fallback_h_step: Optional[float]) -> pd.DataFrame:
    out = agent_df.copy()

    if cmap.decision_frame is not None:
        out["decision_frame_model"] = out[cmap.decision_frame].astype(float).round().astype("Int64")
        return out

    if cmap.decision_h is not None:
        # Nearest h_ext merge.
        h_vals = macro["h_ext"].to_numpy(float)
        frames = macro["frame"].to_numpy(int)
        dec_h = out[cmap.decision_h].to_numpy(float)
        idx = np.searchsorted(h_vals, dec_h)
        idx = np.clip(idx, 0, len(h_vals) - 1)
        idx_prev = np.clip(idx - 1, 0, len(h_vals) - 1)
        choose_prev = np.abs(h_vals[idx_prev] - dec_h) <= np.abs(h_vals[idx] - dec_h)
        idx_final = np.where(choose_prev, idx_prev, idx)
        out["decision_frame_model"] = frames[idx_final]
        return out

    if cmap.step_when_decided is not None and fallback_h_step is not None:
        # Last resort for old fixed-relaxation outputs. Not recommended for variable relaxation.
        out["decision_frame_model"] = np.floor(out[cmap.step_when_decided].astype(float) / fallback_h_step).astype("Int64")
        return out

    raise ValueError(
        "Could not determine decision frame. Need decision_frame, decision_h, or step_when_decided + --fallback_step_to_frame_divisor."
    )


def merge_macro_at_frame(agent_df: pd.DataFrame, macro: pd.DataFrame, frame_col: str, prefix: str) -> pd.DataFrame:
    keys = macro.rename(columns={
        "frame": frame_col,
        "Q_model": f"Q_{prefix}",
        "rho_up_model": f"rho_up_{prefix}",
        "J_out_model": f"J_out_{prefix}",
        "J_out_smooth": f"J_out_smooth_{prefix}",
        "inv_J_out": f"inv_J_out_{prefix}",
        "h_ext": f"h_{prefix}",
    })
    cols = [frame_col, f"h_{prefix}", f"Q_{prefix}", f"rho_up_{prefix}",
            f"J_out_{prefix}", f"J_out_smooth_{prefix}", f"inv_J_out_{prefix}"]
    return agent_df.merge(keys[cols], on=frame_col, how="left")


def estimate_arrival_frames(agent_df: pd.DataFrame, macro: pd.DataFrame, V_col: str, decision_frame_col: str,
                            a1_grid: Iterable[float], use_round: bool = True) -> tuple[pd.DataFrame, float, dict]:
    """Pick a1 by best R2 of tau ~ V + Q_arr/(J_arr+eps). a1 measured in frames per V unit."""
    best = None
    candidates = []
    max_frame = int(macro["frame"].max())
    min_frame = int(macro["frame"].min())

    for a1 in a1_grid:
        tmp = agent_df.copy()
        arr = tmp[decision_frame_col].astype(float) + float(a1) * tmp[V_col].astype(float)
        if use_round:
            arr = np.rint(arr)
        else:
            arr = np.floor(arr)
        arr = np.clip(arr, min_frame, max_frame).astype(int)
        tmp["arrival_frame_est"] = arr
        tmp = merge_macro_at_frame(tmp, macro, "arrival_frame_est", "arr_est")
        tmp["queue_ratio_arr_est"] = tmp["Q_arr_est"] / (tmp["J_out_smooth_arr_est"] + macro.attrs["eps_J"])

        y = tmp["tau_move_model"].to_numpy(float)
        X = tmp[[V_col, "queue_ratio_arr_est"]].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if np.sum(mask) < 10:
            continue
        beta, pred = fit_ols(X[mask], y[mask])
        metrics = regression_metrics(y[mask], pred, n_features=2)
        rec = {"a1_frames_per_V": float(a1), **metrics}
        candidates.append(rec)
        if best is None or metrics["adj_R2"] > best[0]["adj_R2"]:
            best = (rec, tmp, beta)

    if best is None:
        raise RuntimeError("Could not estimate arrival frames with given a1 grid.")

    best_rec, best_df, best_beta = best
    return best_df, float(best_rec["a1_frames_per_V"]), {"best": best_rec, "grid": candidates, "beta": best_beta}


# ============================================================
# Main analysis
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Validate collective bottleneck models for tau_move.")
    parser.add_argument("--agent_csv", required=True, help="tau_move_validation_table_postmean.csv")
    parser.add_argument("--macro_csv", required=True, help="macro_stats.csv")
    parser.add_argument("--output_dir", default="tau_move_bottleneck_results")
    parser.add_argument("--smooth_window", type=int, default=5, help="Rolling window for J_out smoothing in frames")
    parser.add_argument("--eps_J", type=float, default=None, help="Regularization epsilon for 1/(J+eps). Default: 5%% median positive J_smooth")
    parser.add_argument("--fallback_step_to_frame_divisor", type=float, default=None,
                        help="Only for old fixed-step data: frame=floor(step/divisor). Not recommended for variable relaxation.")
    parser.add_argument("--a1_min", type=float, default=0.0, help="Min frames per V for estimated arrival grid")
    parser.add_argument("--a1_max", type=float, default=2.0, help="Max frames per V for estimated arrival grid")
    parser.add_argument("--a1_num", type=int, default=81, help="Number of grid points for estimated arrival a1")
    parser.add_argument("--force_nonnegative", action="store_true", help="Diagnostic: clip regression slopes to nonnegative")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    agent = pd.read_csv(args.agent_csv)
    macro_raw = pd.read_csv(args.macro_csv)
    macro = prepare_macro(macro_raw, args.smooth_window, args.eps_J)
    eps_J = macro.attrs["eps_J"]

    cmap = detect_agent_columns(agent)
    df = agent.copy()
    df["tau_move_model"] = df[cmap.tau].astype(float)
    df = add_decision_frame_if_needed(df, cmap, macro, args.fallback_step_to_frame_divisor)
    df["decision_frame_model"] = df["decision_frame_model"].astype(int)

    # Valid basic data.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["tau_move_model", cmap.V, "decision_frame_model"]).copy()
    df = df[df["tau_move_model"] >= 0].copy()

    # Decision-frame merge.
    df = merge_macro_at_frame(df, macro, "decision_frame_model", "dec")
    df["queue_ratio_dec"] = df["Q_dec"] / (df["J_out_smooth_dec"] + eps_J)

    # Estimated-arrival merge, selecting a1 by grid search.
    a1_grid = np.linspace(args.a1_min, args.a1_max, args.a1_num)
    df_arr, best_a1, arr_info = estimate_arrival_frames(
        df, macro, cmap.V, "decision_frame_model", a1_grid, use_round=True
    )
    # df_arr includes all df columns and arr_est columns. Continue with it.
    df = df_arr

    y = df["tau_move_model"].to_numpy(float)

    models = {
        "B0_V_only": [cmap.V],
        "B1_dec_Q_rho_invJ": [cmap.V, "Q_dec", "rho_up_dec", "inv_J_out_dec"],
        "B2_dec_Q_invJ": [cmap.V, "Q_dec", "inv_J_out_dec"],
        "B3_dec_queue_ratio": [cmap.V, "queue_ratio_dec"],
        "B4_arr_queue_ratio_est": [cmap.V, "queue_ratio_arr_est"],
        "B5_arr_Q_rho_invJ_est": [cmap.V, "Q_arr_est", "rho_up_arr_est", "inv_J_out_arr_est"],
    }

    summary_rows = []
    pred_df = df.copy()

    for name, features in models.items():
        X_df = df[features].replace([np.inf, -np.inf], np.nan)
        tmp = pd.concat([pd.Series(y, name="tau"), X_df], axis=1).dropna()
        if len(tmp) < 10:
            continue
        yy = tmp["tau"].to_numpy(float)
        XX = tmp[features].to_numpy(float)
        beta, pred = fit_ols(XX, yy, nonnegative=args.force_nonnegative)
        metrics = regression_metrics(yy, pred, n_features=len(features))

        row = {"model": name, "features": ",".join(features), **metrics}
        row["intercept"] = beta[0]
        for f, b in zip(features, beta[1:]):
            row[f"coef_{f}"] = b
        summary_rows.append(row)

        # Store predictions aligned to original index.
        full_pred = np.full(len(df), np.nan)
        X_full = X_df.to_numpy(float)
        mask = np.all(np.isfinite(X_full), axis=1) & np.isfinite(y)
        full_pred[mask] = np.column_stack([np.ones(np.sum(mask)), X_full[mask]]) @ beta
        pred_col = f"pred_{name}"
        pred_df[pred_col] = full_pred

        plot_pred(
            pred_df["tau_move_model"].to_numpy(float),
            pred_df[pred_col].to_numpy(float),
            f"tau_move bottleneck validation: {name}",
            f"Predicted {name}",
            os.path.join(args.output_dir, f"pred_{name}_linear.png"),
            os.path.join(args.output_dir, f"pred_{name}_loglog.png"),
        )

    summary = pd.DataFrame(summary_rows).sort_values("adj_R2", ascending=False)
    summary.to_csv(os.path.join(args.output_dir, "tau_move_bottleneck_models_summary.csv"), index=False)

    # Arrival grid summary.
    pd.DataFrame(arr_info["grid"]).to_csv(os.path.join(args.output_dir, "arrival_a1_grid_search.csv"), index=False)

    # Save merged and predictions.
    pred_df.to_csv(os.path.join(args.output_dir, "tau_move_bottleneck_merged_predictions.csv"), index=False)

    # Component panels.
    component_cols = [
        (cmap.V, r"$V_i^{dec}$"),
        ("Q_dec", r"$Q(t_i^{dec})$"),
        ("rho_up_dec", r"$\rho_{up}(t_i^{dec})$"),
        ("inv_J_out_dec", r"$1/(J_{out}(t_i^{dec})+\epsilon)$"),
        ("queue_ratio_dec", r"$Q(t_i^{dec})/(J_{out}+\epsilon)$"),
        ("queue_ratio_arr_est", r"$Q(t_i^{arr,est})/(J_{out}+\epsilon)$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, component_cols):
        if col not in pred_df.columns:
            ax.axis("off")
            continue
        ax.scatter(pred_df[col], pred_df["tau_move_model"], alpha=0.5, s=12)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\tau_i^{move}$")
        ax.set_title(f"{col} vs tau_move")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "bottleneck_components_vs_tau_move.png"), dpi=200)
    plt.close(fig)

    # Comparison plot for best models.
    best_names = summary["model"].head(4).tolist() if len(summary) > 0 else []
    if best_names:
        fig, axes = plt.subplots(1, len(best_names), figsize=(6 * len(best_names), 5))
        if len(best_names) == 1:
            axes = [axes]
        for ax, name in zip(axes, best_names):
            col = f"pred_{name}"
            ax.scatter(pred_df[col], pred_df["tau_move_model"], alpha=0.5, s=12)
            mn = np.nanmin([pred_df[col].min(), pred_df["tau_move_model"].min(), 0])
            mx = np.nanmax([pred_df[col].max(), pred_df["tau_move_model"].max()])
            ax.plot([mn, mx], [mn, mx], linestyle="--")
            ax.set_title(name)
            ax.set_xlabel("predicted")
            ax.set_ylabel("observed tau_move")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "best_bottleneck_model_comparison.png"), dpi=200)
        plt.close(fig)

    # Text summary.
    with open(os.path.join(args.output_dir, "analysis_summary_bottleneck.txt"), "w", encoding="utf-8") as f:
        f.write("=== tau_move collective bottleneck validation ===\n")
        f.write(f"agent_csv: {args.agent_csv}\n")
        f.write(f"macro_csv: {args.macro_csv}\n")
        f.write(f"n_valid: {len(df)}\n")
        f.write(f"eps_J: {eps_J}\n")
        f.write(f"smooth_window: {args.smooth_window}\n")
        f.write(f"best_a1_frames_per_V_for_arrival_est: {best_a1}\n\n")
        f.write("Columns used:\n")
        f.write(f"  tau_move: {cmap.tau}\n")
        f.write(f"  V_at_decision: {cmap.V}\n")
        f.write(f"  decision_frame source: {cmap.decision_frame or cmap.decision_h or cmap.step_when_decided}\n\n")

        f.write("Single-variable correlations with tau_move:\n")
        for col in [cmap.V, "Q_dec", "rho_up_dec", "inv_J_out_dec", "queue_ratio_dec", "queue_ratio_arr_est"]:
            if col in pred_df.columns:
                f.write(f"  corr({col}, tau_move) pearson = {safe_corr(pred_df[col], pred_df['tau_move_model'], 'pearson'):.6f}\n")
                f.write(f"  corr({col}, tau_move) spearman = {safe_corr(pred_df[col], pred_df['tau_move_model'], 'spearman'):.6f}\n")
        f.write("\nModel summary sorted by adjusted R2:\n")
        for _, r in summary.iterrows():
            f.write(
                f"  {r['model']}: adj_R2={r['adj_R2']:.6f}, R2={r['R2']:.6f}, "
                f"pearson={r['pearson']:.6f}, spearman={r['spearman']:.6f}, "
                f"pearson_log={r['pearson_log']:.6f}, spearman_log={r['spearman_log']:.6f}\n"
            )
            f.write(f"     features: {r['features']}\n")
            coef_parts = [f"intercept={r.get('intercept', np.nan):.6f}"]
            for c in summary.columns:
                if c.startswith("coef_") and pd.notna(r.get(c, np.nan)):
                    coef_parts.append(f"{c.replace('coef_', '')}={r[c]:.6f}")
            f.write("     coefficients: " + ", ".join(coef_parts) + "\n")

        f.write("\nArrival a1 grid best:\n")
        f.write(str(arr_info["best"]) + "\n")

        f.write("\nSaved files:\n")
        for name in sorted(os.listdir(args.output_dir)):
            f.write(f"  - {name}\n")

    print(f"[DONE] Results saved to: {args.output_dir}")
    if len(summary) > 0:
        print(summary[["model", "adj_R2", "R2", "pearson", "spearman", "pearson_log", "spearman_log"]].to_string(index=False))
        print(f"best_a1_frames_per_V_for_arrival_est = {best_a1}")
        print(f"eps_J = {eps_J}")


if __name__ == "__main__":
    main()
