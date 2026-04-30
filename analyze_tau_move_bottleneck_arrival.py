#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate tau_move using actual upstream-arrival information and bottleneck IDs.

This script is designed for outputs from:
  simulate_avalanche_priorityC_postmean_varh_arrival_v2.py

It uses actual recorded values:
  - arrival_step_upstream
  - arrival_frame_upstream
  - arrival_h_upstream
  - bottleneck_id

and validates models such as:
  tau_move ~ V_dec + Q(t_dec) + rho_up(t_dec) + 1/(J_out(t_dec)+eps)
  tau_move ~ V_dec + Q(t_arr)/J_out(t_arr)
  tau_move ~ V_dec + Q_b(t_arr)/J_b(t_arr)   [if bottleneck_id and exit_events are available]

Inputs can be given explicitly or through --results_dir.

Typical usage:
  python analyze_tau_move_actual_arrival_bottleneck.py \
    --results_dir simulation_results_threshold_10m_c_postmean_arrival \
    --output_dir tau_move_actual_arrival_bottleneck_results

or:
  python analyze_tau_move_actual_arrival_bottleneck.py \
    --agent_csv simulation_results_threshold_10m_c_postmean_arrival/agents_frame_118.csv \
    --macro_csv simulation_results_threshold_10m_c_postmean_arrival/macro_stats.csv \
    --exit_events_csv simulation_results_threshold_10m_c_postmean_arrival/exit_events.csv \
    --output_dir tau_move_actual_arrival_bottleneck_results
"""

from __future__ import annotations

import argparse
import glob
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
        raise ValueError(
            f"None of these columns were found: {list(candidates)}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def latest_agents_frame(results_dir: str) -> str:
    files = glob.glob(os.path.join(results_dir, "agents_frame_*.csv"))
    if not files:
        raise FileNotFoundError(f"No agents_frame_*.csv found in {results_dir}")

    def frame_number(path: str) -> int:
        m = re.search(r"agents_frame_(\d+)\.csv$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(files, key=frame_number)


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    tmp = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 3:
        return np.nan
    if tmp.iloc[:, 0].nunique() < 2 or tmp.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method=method))


def fit_ols(X: np.ndarray, y: np.ndarray, nonnegative: bool = False):
    """OLS with intercept. Optional nonnegative slopes by simple post-clipping diagnostic."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    if nonnegative:
        beta[1:] = np.maximum(beta[1:], 0.0)
        beta[0] = np.mean(y - X @ beta[1:])

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
        return dict(n=n, R2=np.nan, adj_R2=np.nan, rmse=np.nan, mae=np.nan,
                    pearson=np.nan, spearman=np.nan, pearson_log=np.nan, spearman_log=np.nan)

    resid = y - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - n_features - 1, 1) if np.isfinite(r2) else np.nan
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    pearson = safe_corr(pd.Series(y_pred), pd.Series(y), "pearson")
    spearman = safe_corr(pd.Series(y_pred), pd.Series(y), "spearman")

    pos = (y > 0) & (y_pred > 0)
    if np.sum(pos) >= 3:
        pearson_log = safe_corr(pd.Series(np.log(y_pred[pos])), pd.Series(np.log(y[pos])), "pearson")
        spearman_log = safe_corr(pd.Series(np.log(y_pred[pos])), pd.Series(np.log(y[pos])), "spearman")
    else:
        pearson_log = np.nan
        spearman_log = np.nan

    return dict(n=n, R2=r2, adj_R2=adj_r2, rmse=rmse, mae=mae,
                pearson=pearson, spearman=spearman,
                pearson_log=pearson_log, spearman_log=spearman_log)


def smooth_series(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values.astype(float)
    return values.astype(float).rolling(window=window, center=True, min_periods=1).mean()


def plot_pred(y: np.ndarray, y_pred: np.ndarray, title: str, xlabel: str, out_linear: str, out_log: str) -> None:
    mask = np.isfinite(y) & np.isfinite(y_pred)
    yy = y[mask]
    pp = y_pred[mask]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pp, yy, alpha=0.5, s=14)
    if len(yy) > 0:
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
    ax.scatter(y_pred[pos], y[pos], alpha=0.5, s=14)
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


@dataclass
class AgentColumns:
    tau: str
    V_dec: str
    dec_frame: str
    arr_frame: str
    bottleneck_id: Optional[str]
    evac_frame: Optional[str]


# ============================================================
# Data preparation
# ============================================================

def prepare_macro(macro_raw: pd.DataFrame, smooth_window: int, eps_J: Optional[float]) -> pd.DataFrame:
    m = macro_raw.copy()
    if "frame" not in m.columns:
        m["frame"] = np.arange(len(m), dtype=int)
    if "h_ext" not in m.columns:
        raise ValueError("macro_stats.csv needs h_ext column.")

    Q_col = first_existing_column(m, ["Q_transport", "Q", "queue", "transport_inventory", "q_transport"], required=False)
    if Q_col is None:
        if "m_dec_cum" in m.columns and "m_evac_cum" in m.columns:
            m["Q_transport"] = m["m_dec_cum"] - m["m_evac_cum"]
            Q_col = "Q_transport"
        elif "m_dec_cum" in m.columns and "m_evac" in m.columns:
            m["Q_transport"] = m["m_dec_cum"] - m["m_evac"]
            Q_col = "Q_transport"
        else:
            raise ValueError("Could not find or construct Q_transport.")
    m["Q_global"] = m[Q_col].astype(float)

    rho_col = first_existing_column(m, ["rho_up", "rho_upstream", "upstream_density"], required=False)
    m["rho_up_global"] = m[rho_col].astype(float) if rho_col is not None else np.nan

    J_col = first_existing_column(
        m,
        ["J_out_frame", "service_rate_event_frame", "exit_count_frame", "J_out", "J_out_per_step"],
        required=False,
    )
    if J_col is None:
        if "m_evac_cum" in m.columns:
            m["J_out_frame"] = m["m_evac_cum"].diff().fillna(m["m_evac_cum"]).clip(lower=0)
            J_col = "J_out_frame"
        elif "m_evac" in m.columns:
            m["J_out_frame"] = m["m_evac"].diff().fillna(m["m_evac"]).clip(lower=0)
            J_col = "J_out_frame"
        else:
            raise ValueError("Could not find or construct J_out.")
    m["J_global"] = m[J_col].astype(float)
    m["J_global_smooth"] = smooth_series(m["J_global"], smooth_window)

    if eps_J is None:
        positive = m.loc[m["J_global_smooth"] > 0, "J_global_smooth"]
        eps_val = max(1e-9, 0.05 * float(positive.median())) if len(positive) else 1.0
    else:
        eps_val = float(eps_J)
    m.attrs["eps_J"] = eps_val
    m["inv_J_global"] = 1.0 / (m["J_global_smooth"] + eps_val)

    return m[["frame", "h_ext", "Q_global", "rho_up_global", "J_global", "J_global_smooth", "inv_J_global"]]


def prepare_agent(agent_raw: pd.DataFrame) -> tuple[pd.DataFrame, AgentColumns]:
    a = agent_raw.copy().replace([np.inf, -np.inf], np.nan)

    V_col = first_existing_column(a, ["V_at_decision", "V_dec", "V_i_dec"])
    dec_frame_col = first_existing_column(a, ["decision_frame", "frame_decision", "dec_frame"])
    arr_frame_col = first_existing_column(a, ["arrival_frame_upstream", "arrival_frame", "arr_frame"])
    bottleneck_col = first_existing_column(a, ["bottleneck_id", "arrival_bottleneck_id", "b_id"], required=False)
    evac_frame_col = first_existing_column(a, ["evacuation_frame", "evac_frame"], required=False)

    tau_col = first_existing_column(a, ["tau_move", "tau_move_steps", "movement_delay"], required=False)
    if tau_col is None:
        step_dec = first_existing_column(a, ["step_when_decided", "decision_step"])
        step_evac = first_existing_column(a, ["evacuation_time", "evacuation_step"])
        a["tau_move"] = a[step_evac].astype(float) - a[step_dec].astype(float)
        tau_col = "tau_move"

    # Standardized columns.
    a["tau_move_model"] = a[tau_col].astype(float)
    a["V_dec_model"] = a[V_col].astype(float)
    a["decision_frame_model"] = a[dec_frame_col].astype(float).round()
    a["arrival_frame_model"] = a[arr_frame_col].astype(float).round()
    if bottleneck_col is not None:
        a["bottleneck_id_model"] = a[bottleneck_col]
    else:
        a["bottleneck_id_model"] = np.nan

    # Filter valid decided, arrived, evacuated agents.
    a = a.dropna(subset=["tau_move_model", "V_dec_model", "decision_frame_model", "arrival_frame_model"]).copy()
    a = a[(a["tau_move_model"] >= 0) & (a["decision_frame_model"] >= 0) & (a["arrival_frame_model"] >= 0)].copy()
    a["decision_frame_model"] = a["decision_frame_model"].astype(int)
    a["arrival_frame_model"] = a["arrival_frame_model"].astype(int)

    if evac_frame_col is not None:
        a["evacuation_frame_model"] = a[evac_frame_col].astype(float).round()
    else:
        a["evacuation_frame_model"] = np.nan

    cols = AgentColumns(tau_col, V_col, dec_frame_col, arr_frame_col, bottleneck_col, evac_frame_col)
    return a, cols


def merge_global_at_frame(agent: pd.DataFrame, macro: pd.DataFrame, frame_col: str, prefix: str) -> pd.DataFrame:
    key = macro.rename(columns={
        "frame": frame_col,
        "h_ext": f"h_{prefix}",
        "Q_global": f"Q_{prefix}",
        "rho_up_global": f"rho_up_{prefix}",
        "J_global": f"J_{prefix}",
        "J_global_smooth": f"J_smooth_{prefix}",
        "inv_J_global": f"inv_J_{prefix}",
    })
    cols = [frame_col, f"h_{prefix}", f"Q_{prefix}", f"rho_up_{prefix}", f"J_{prefix}", f"J_smooth_{prefix}", f"inv_J_{prefix}"]
    return agent.merge(key[cols], on=frame_col, how="left")


def prepare_exit_events(exit_csv: Optional[str], smooth_window: int, eps_J: float) -> Optional[pd.DataFrame]:
    if exit_csv is None or not os.path.exists(exit_csv):
        return None
    e = pd.read_csv(exit_csv)
    if "frame" not in e.columns:
        return None
    if "bottleneck_id" not in e.columns:
        return None
    e = e.dropna(subset=["frame", "bottleneck_id"]).copy()
    e = e[e["bottleneck_id"] >= 0].copy()
    if len(e) == 0:
        return None
    e["frame"] = e["frame"].astype(int)
    e["bottleneck_id"] = e["bottleneck_id"].astype(int)
    counts = e.groupby(["bottleneck_id", "frame"]).size().reset_index(name="J_b")

    # Fill missing frames for each bottleneck.
    frames = np.arange(int(e["frame"].min()), int(e["frame"].max()) + 1)
    all_rows = []
    for b, sub in counts.groupby("bottleneck_id"):
        tmp = pd.DataFrame({"frame": frames})
        tmp["bottleneck_id"] = int(b)
        tmp = tmp.merge(sub[["frame", "J_b"]], on="frame", how="left")
        tmp["J_b"] = tmp["J_b"].fillna(0.0)
        tmp["J_b_smooth"] = smooth_series(tmp["J_b"], smooth_window)
        tmp["inv_J_b"] = 1.0 / (tmp["J_b_smooth"] + eps_J)
        all_rows.append(tmp)
    return pd.concat(all_rows, ignore_index=True)


def add_bottleneck_queue(agent: pd.DataFrame, frames: np.ndarray) -> pd.DataFrame:
    """Compute Q_b_arr: count agents in same bottleneck that have arrived but not evacuated at arrival frame.

    Requires bottleneck_id_model, arrival_frame_model, and ideally evacuation_frame_model.
    If evacuation_frame_model is missing, uses global tau_move endpoint poorly; Q_b will be skipped.
    """
    out = agent.copy()
    if out["bottleneck_id_model"].isna().all():
        return out
    if "evacuation_frame_model" not in out.columns or out["evacuation_frame_model"].isna().all():
        return out

    out["Q_b_arr"] = np.nan
    out["Q_b_dec"] = np.nan
    valid = out.dropna(subset=["bottleneck_id_model", "arrival_frame_model", "decision_frame_model", "evacuation_frame_model"]).copy()
    valid = valid[valid["bottleneck_id_model"] >= 0]
    if len(valid) == 0:
        return out

    # O(B * F * n_b) for n~2000 and F~100-400 is fine.
    q_arr = pd.Series(index=out.index, dtype=float)
    q_dec = pd.Series(index=out.index, dtype=float)
    for b, sub in valid.groupby("bottleneck_id_model"):
        idx = sub.index.to_numpy()
        arr = sub["arrival_frame_model"].to_numpy(int)
        dec = sub["decision_frame_model"].to_numpy(int)
        evac = sub["evacuation_frame_model"].to_numpy(float)
        for local_idx, global_idx in enumerate(idx):
            fa = arr[local_idx]
            fd = dec[local_idx]
            q_arr.loc[global_idx] = np.sum((arr <= fa) & (evac > fa))
            q_dec.loc[global_idx] = np.sum((arr <= fd) & (evac > fd))

    out.loc[q_arr.index, "Q_b_arr"] = q_arr
    out.loc[q_dec.index, "Q_b_dec"] = q_dec
    return out


def merge_bottleneck_J(agent: pd.DataFrame, Jb: Optional[pd.DataFrame], frame_col: str, prefix: str) -> pd.DataFrame:
    if Jb is None or agent["bottleneck_id_model"].isna().all():
        return agent
    out = agent.copy()
    out["bottleneck_id_model"] = out["bottleneck_id_model"].astype(float)
    key = Jb.rename(columns={
        "frame": frame_col,
        "J_b": f"J_b_{prefix}",
        "J_b_smooth": f"J_b_smooth_{prefix}",
        "inv_J_b": f"inv_J_b_{prefix}",
    })
    cols = ["bottleneck_id", frame_col, f"J_b_{prefix}", f"J_b_smooth_{prefix}", f"inv_J_b_{prefix}"]
    key = key[cols].rename(columns={"bottleneck_id": "bottleneck_id_model"})
    key["bottleneck_id_model"] = key["bottleneck_id_model"].astype(float)
    return out.merge(key, on=["bottleneck_id_model", frame_col], how="left")


# ============================================================
# Main analysis
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate tau_move using actual upstream arrival and bottleneck IDs.")
    parser.add_argument("--results_dir", default=None, help="Directory containing macro_stats.csv, exit_events.csv, agents_frame_*.csv")
    parser.add_argument("--agent_csv", default=None, help="agents_frame_*.csv or tau validation csv with actual arrival columns")
    parser.add_argument("--macro_csv", default=None, help="macro_stats.csv")
    parser.add_argument("--exit_events_csv", default=None, help="exit_events.csv with bottleneck_id, optional")
    parser.add_argument("--output_dir", default="tau_move_actual_arrival_bottleneck_results")
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--eps_J", type=float, default=None)
    parser.add_argument("--force_nonnegative", action="store_true")
    args = parser.parse_args()

    if args.results_dir is not None:
        if args.macro_csv is None:
            args.macro_csv = os.path.join(args.results_dir, "macro_stats.csv")
        if args.exit_events_csv is None:
            cand = os.path.join(args.results_dir, "exit_events.csv")
            args.exit_events_csv = cand if os.path.exists(cand) else None
        if args.agent_csv is None:
            args.agent_csv = latest_agents_frame(args.results_dir)

    if args.agent_csv is None or args.macro_csv is None:
        raise ValueError("Need --results_dir or both --agent_csv and --macro_csv")

    ensure_dir(args.output_dir)
    raw_agent = pd.read_csv(args.agent_csv)
    raw_macro = pd.read_csv(args.macro_csv)

    macro = prepare_macro(raw_macro, smooth_window=args.smooth_window, eps_J=args.eps_J)
    eps_J = float(macro.attrs["eps_J"])
    agent, cols = prepare_agent(raw_agent)

    # Merge global macro quantities at decision and actual arrival frames.
    df = merge_global_at_frame(agent, macro, "decision_frame_model", "dec")
    df = merge_global_at_frame(df, macro, "arrival_frame_model", "arr")
    df["queue_ratio_dec"] = df["Q_dec"] / (df["J_smooth_dec"] + eps_J)
    df["queue_ratio_arr"] = df["Q_arr"] / (df["J_smooth_arr"] + eps_J)

    # Bottleneck-specific quantities if possible.
    Jb = prepare_exit_events(args.exit_events_csv, smooth_window=args.smooth_window, eps_J=eps_J)
    df = add_bottleneck_queue(df, macro["frame"].to_numpy(int))
    df = merge_bottleneck_J(df, Jb, "arrival_frame_model", "arr")
    df = merge_bottleneck_J(df, Jb, "decision_frame_model", "dec")

    if "Q_b_arr" in df.columns and "J_b_smooth_arr" in df.columns:
        df["queue_ratio_b_arr"] = df["Q_b_arr"] / (df["J_b_smooth_arr"] + eps_J)
    if "Q_b_dec" in df.columns and "J_b_smooth_dec" in df.columns:
        df["queue_ratio_b_dec"] = df["Q_b_dec"] / (df["J_b_smooth_dec"] + eps_J)

    df = df.replace([np.inf, -np.inf], np.nan)
    y = df["tau_move_model"].to_numpy(float)

    models: dict[str, list[str]] = {
        "A0_V_only": ["V_dec_model"],
        "A1_dec_Q_rho_invJ": ["V_dec_model", "Q_dec", "rho_up_dec", "inv_J_dec"],
        "A2_dec_queue_ratio": ["V_dec_model", "queue_ratio_dec"],
        "A3_arr_Q_rho_invJ_actual": ["V_dec_model", "Q_arr", "rho_up_arr", "inv_J_arr"],
        "A4_arr_queue_ratio_actual": ["V_dec_model", "queue_ratio_arr"],
    }

    # Optional bottleneck-specific models.
    if "queue_ratio_b_arr" in df.columns:
        models["B1_bottleneck_arr_queue_ratio"] = ["V_dec_model", "queue_ratio_b_arr"]
    if "Q_b_arr" in df.columns and "inv_J_b_arr" in df.columns:
        models["B2_bottleneck_arr_Q_invJ"] = ["V_dec_model", "Q_b_arr", "inv_J_b_arr"]
    if "queue_ratio_b_dec" in df.columns:
        models["B3_bottleneck_dec_queue_ratio"] = ["V_dec_model", "queue_ratio_b_dec"]

    summary_rows = []
    pred_df = df.copy()

    for name, features in models.items():
        missing = [c for c in features if c not in pred_df.columns]
        if missing:
            continue
        X_df = pred_df[features].replace([np.inf, -np.inf], np.nan)
        tmp = pd.concat([pred_df["tau_move_model"], X_df], axis=1).dropna()
        if len(tmp) < 10:
            continue
        yy = tmp["tau_move_model"].to_numpy(float)
        XX = tmp[features].to_numpy(float)
        beta, pred = fit_ols(XX, yy, nonnegative=args.force_nonnegative)
        metrics = regression_metrics(yy, pred, n_features=len(features))
        row = {"model": name, "features": ",".join(features), **metrics, "intercept": beta[0]}
        for f, b in zip(features, beta[1:]):
            row[f"coef_{f}"] = b
        summary_rows.append(row)

        # Full prediction column aligned to pred_df.
        full_pred = np.full(len(pred_df), np.nan)
        X_full = X_df.to_numpy(float)
        mask = np.isfinite(pred_df["tau_move_model"].to_numpy(float)) & np.all(np.isfinite(X_full), axis=1)
        full_pred[mask] = np.column_stack([np.ones(np.sum(mask)), X_full[mask]]) @ beta
        pred_col = f"pred_{name}"
        pred_df[pred_col] = full_pred

        plot_pred(
            pred_df["tau_move_model"].to_numpy(float),
            pred_df[pred_col].to_numpy(float),
            f"tau_move actual-arrival bottleneck validation: {name}",
            f"Predicted {name}",
            os.path.join(args.output_dir, f"pred_{name}_linear.png"),
            os.path.join(args.output_dir, f"pred_{name}_loglog.png"),
        )

    summary = pd.DataFrame(summary_rows).sort_values("adj_R2", ascending=False)
    summary.to_csv(os.path.join(args.output_dir, "tau_move_actual_arrival_bottleneck_models_summary.csv"), index=False)
    pred_df.to_csv(os.path.join(args.output_dir, "tau_move_actual_arrival_bottleneck_merged_predictions.csv"), index=False)

    # Component panels.
    component_cols = [
        ("V_dec_model", r"$V_i^{dec}$"),
        ("Q_dec", r"$Q(t_i^{dec})$"),
        ("Q_arr", r"$Q(t_i^{arr})$"),
        ("rho_up_dec", r"$\rho_{up}(t_i^{dec})$"),
        ("rho_up_arr", r"$\rho_{up}(t_i^{arr})$"),
        ("queue_ratio_dec", r"$Q(t_i^{dec})/(J_{out}+\epsilon)$"),
        ("queue_ratio_arr", r"$Q(t_i^{arr})/(J_{out}+\epsilon)$"),
        ("queue_ratio_b_arr", r"$Q_b(t_i^{arr})/(J_b+\epsilon)$"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, component_cols):
        if col not in pred_df.columns or pred_df[col].notna().sum() < 3:
            ax.axis("off")
            ax.set_title(f"{col}: unavailable")
            continue
        ax.scatter(pred_df[col], pred_df["tau_move_model"], alpha=0.5, s=12)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\tau_i^{move}$")
        ax.set_title(f"{col} vs tau_move")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "actual_arrival_bottleneck_components_vs_tau_move.png"), dpi=200)
    plt.close(fig)

    # Best model comparison.
    best_names = summary["model"].head(4).tolist() if len(summary) else []
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
        fig.savefig(os.path.join(args.output_dir, "best_actual_arrival_bottleneck_model_comparison.png"), dpi=200)
        plt.close(fig)

    # Text summary.
    with open(os.path.join(args.output_dir, "analysis_summary_actual_arrival_bottleneck.txt"), "w", encoding="utf-8") as f:
        f.write("=== tau_move actual-arrival collective bottleneck validation ===\n")
        f.write(f"agent_csv: {args.agent_csv}\n")
        f.write(f"macro_csv: {args.macro_csv}\n")
        f.write(f"exit_events_csv: {args.exit_events_csv}\n")
        f.write(f"n_valid: {len(pred_df)}\n")
        f.write(f"eps_J: {eps_J}\n")
        f.write(f"smooth_window: {args.smooth_window}\n\n")

        f.write("Columns detected:\n")
        f.write(f"  tau_move: {cols.tau}\n")
        f.write(f"  V_at_decision: {cols.V_dec}\n")
        f.write(f"  decision_frame: {cols.dec_frame}\n")
        f.write(f"  arrival_frame_upstream: {cols.arr_frame}\n")
        f.write(f"  bottleneck_id: {cols.bottleneck_id}\n")
        f.write(f"  evacuation_frame: {cols.evac_frame}\n\n")

        f.write("Single-variable correlations with tau_move:\n")
        for col in ["V_dec_model", "Q_dec", "Q_arr", "rho_up_dec", "rho_up_arr",
                    "inv_J_dec", "inv_J_arr", "queue_ratio_dec", "queue_ratio_arr",
                    "Q_b_arr", "inv_J_b_arr", "queue_ratio_b_arr"]:
            if col in pred_df.columns and pred_df[col].notna().sum() >= 3:
                f.write(f"  corr({col}, tau_move) pearson = {safe_corr(pred_df[col], pred_df['tau_move_model'], 'pearson'):.6f}\n")
                f.write(f"  corr({col}, tau_move) spearman = {safe_corr(pred_df[col], pred_df['tau_move_model'], 'spearman'):.6f}\n")

        f.write("\nModel summary sorted by adjusted R2:\n")
        for _, r in summary.iterrows():
            f.write(
                f"  {r['model']}: adj_R2={r['adj_R2']:.6f}, R2={r['R2']:.6f}, "
                f"pearson={r['pearson']:.6f}, spearman={r['spearman']:.6f}, "
                f"pearson_log={r['pearson_log']:.6f}, spearman_log={r['spearman_log']:.6f}, "
                f"rmse={r['rmse']:.3f}, mae={r['mae']:.3f}\n"
            )
            f.write(f"     features: {r['features']}\n")
            coef_parts = [f"intercept={r.get('intercept', np.nan):.6f}"]
            for c in summary.columns:
                if c.startswith("coef_") and pd.notna(r.get(c, np.nan)):
                    coef_parts.append(f"{c.replace('coef_', '')}={r[c]:.6f}")
            f.write("     coefficients: " + ", ".join(coef_parts) + "\n")

        f.write("\nSaved files:\n")
        for name in sorted(os.listdir(args.output_dir)):
            f.write(f"  - {name}\n")

    print(f"[DONE] Results saved to: {args.output_dir}")
    if len(summary):
        print(summary[["model", "adj_R2", "R2", "pearson", "spearman", "pearson_log", "spearman_log"]].to_string(index=False))
    print(f"eps_J = {eps_J}")


if __name__ == "__main__":
    main()
