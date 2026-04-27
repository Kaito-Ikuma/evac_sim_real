import os
import re
import glob
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# base_map から地形拘束指標 V(x,y) を再構成するために使う
import skfmm


# ============================================================
# 0. 基本設定
# ============================================================

@dataclass
class Config:
    relaxation_steps: int = 5000
    h_step: float = 0.005
    late_quantile: float = 0.90
    dense_quantile: float = 0.90


# ============================================================
# 1. ファイル読み込み
# ============================================================

def extract_frame_number(path: str):
    m = re.search(r"agents_frame_(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None


def load_agents_panel(results_dir: str) -> pd.DataFrame:
    """
    agents_frame_*.csv をすべて読み込み、
    1行 = (agent_id, frame) の panel DataFrame を作る。

    ここで 'frame' を追加して時系列解析できる形にする。
    """
    files = sorted(glob.glob(os.path.join(results_dir, "agents_frame_*.csv")))
    if not files:
        raise FileNotFoundError(f"agents_frame_*.csv が見つかりません: {results_dir}")

    dfs = []
    for f in files:
        frame = extract_frame_number(f)
        if frame is None:
            continue
        df = pd.read_csv(f)
        df["frame"] = frame
        dfs.append(df)

    if not dfs:
        raise RuntimeError("有効な agents_frame_*.csv を読み込めませんでした。")

    panel = pd.concat(dfs, ignore_index=True)
    panel = panel.sort_values(["agent_id", "frame"]).reset_index(drop=True)
    return panel


def load_base_map(base_map_csv: str):
    """
    base_map_potential_real_10m.csv を読み込み、
    シミュレーション本体と同様に GRID_W, GRID_H と
    FMM 距離ポテンシャル V_field を再構成する。

    これにより各フレーム位置 (x,y) から
    「安全圏までの地形拘束距離」に相当する V(x,y) を参照できる。
    """
    if not os.path.exists(base_map_csv):
        raise FileNotFoundError(f"base_map_csv が見つかりません: {base_map_csv}")

    df = pd.read_csv(base_map_csv)

    required = ["lon", "lat", "potential_v", "is_obstacle"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"base_map_csv に必要列がありません: {missing}")

    lon_unique = np.sort(df["lon"].unique())
    lat_unique = np.sort(df["lat"].unique())
    grid_w = len(lon_unique)
    grid_h = len(lat_unique)

    # 初期 potential_v を配置
    base_potential = np.zeros((grid_h, grid_w))
    obstacles = np.zeros((grid_h, grid_w), dtype=bool)

    lon_to_ix = {v: i for i, v in enumerate(lon_unique)}
    lat_to_iy = {v: i for i, v in enumerate(lat_unique)}

    for _, row in df.iterrows():
        x = lon_to_ix[row["lon"]]
        y = lat_to_iy[row["lat"]]
        base_potential[y, x] = row["potential_v"]
        obstacles[y, x] = bool(row["is_obstacle"])

    # シミュレーション本体と同様に safe set を potential_v == 0 で定義
    phi = np.ones((grid_h, grid_w))
    phi[base_potential == 0] = 0

    phi_masked = np.ma.MaskedArray(phi, mask=obstacles)
    distance_field = skfmm.distance(phi_masked).data
    distance_field[obstacles] = np.inf

    return {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "V_field": distance_field,
        "obstacles": obstacles,
    }


# ============================================================
# 2. 時系列列の追加
# ============================================================

def add_time_columns(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    frame から外場 h_ext(frame) を追加し、
    step_when_decided / evacuation_time から
    decision_frame / evacuation_frame を再構成する。

    ここでの考え方:
      - 絶対step index を RELAXATION_STEPS で割ると
        そのイベントが起きたフレーム番号になる
    """
    out = panel.copy()

    out["h_ext"] = out["frame"] * cfg.h_step

    out["decision_frame"] = np.where(
        out["step_when_decided"] >= 0,
        (out["step_when_decided"] // cfg.relaxation_steps).astype("Int64"),
        pd.NA
    )

    out["evacuation_frame"] = np.where(
        out["evacuation_time"] >= 0,
        (out["evacuation_time"] // cfg.relaxation_steps).astype("Int64"),
        pd.NA
    )

    return out


def add_motion_columns(panel: pd.DataFrame, V_field: np.ndarray) -> pd.DataFrame:
    """
    各 agent の frame-to-frame 変化量を追加する。

    追加する主な列:
      V_pos       : 現在位置での FMM距離ポテンシャル
      prev_x,prev_y
      dx_frame, dy_frame
      disp_frame  : フレーム間移動量
      dV_frame    : 1フレームでどれだけ安全圏へ近づいたか
      no_progress : V が減っていない (地形拘束 or 渋滞拘束) 指標
    """
    out = panel.copy()

    # 現在位置の地形拘束距離 V(x,y)
    # x,y は float 保存でも格子 index として使われているので int 化
    xi = out["x"].astype(int).to_numpy()
    yi = out["y"].astype(int).to_numpy()
    out["V_pos"] = V_field[yi, xi]

    # frame-to-frame 差分
    out["prev_x"] = out.groupby("agent_id")["x"].shift(1)
    out["prev_y"] = out.groupby("agent_id")["y"].shift(1)
    out["prev_V"] = out.groupby("agent_id")["V_pos"].shift(1)
    out["prev_s"] = out.groupby("agent_id")["s"].shift(1)

    out["dx_frame"] = out["x"] - out["prev_x"]
    out["dy_frame"] = out["y"] - out["prev_y"]
    out["disp_frame"] = np.sqrt(out["dx_frame"]**2 + out["dy_frame"]**2)

    # V が減るほど安全圏に近づいた
    out["dV_frame"] = out["prev_V"] - out["V_pos"]

    # no_progress = 位置変化が小さい or V が減っていない
    out["stagnated"] = out["disp_frame"].fillna(0.0) < 1e-12
    out["no_progress"] = out["dV_frame"].fillna(0.0) <= 0.0

    return out


# ============================================================
# 3. 決断前 / 決断後 / 滞留 の集約量
# ============================================================

def first_nonnegative(series):
    s = series[series >= 0]
    if len(s) == 0:
        return np.nan
    return s.iloc[0]

def last_nonnegative(series):
    s = series[series >= 0]
    if len(s) == 0:
        return np.nan
    return s.iloc[-1]

def summarize_per_agent(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    1 agent ごとに時系列を集約して summary table を作る。

    3つの論点に対応して量を作る:
      (A) 決断前の心理・社会的形成
      (B) 決断後の混雑・地形拘束
      (C) 空間的滞留
    """
    records = []

    # 高密度閾値は global before density の上位分位点で定義
    dense_threshold = panel["local_density_before_move"].quantile(cfg.dense_quantile)

    for agent_id, g in panel.groupby("agent_id"):
        g = g.sort_values("frame").reset_index(drop=True)

        phi = g["phi"].iloc[0]
        mu = g["mu"].iloc[0]

        step_when_decided = first_nonnegative(g["step_when_decided"])
        evacuation_time = first_nonnegative(g["evacuation_time"])

        decision_frame = g["decision_frame"].dropna()
        evacuation_frame = g["evacuation_frame"].dropna()

        decision_frame = int(decision_frame.iloc[0]) if len(decision_frame) > 0 else None
        evacuation_frame = int(evacuation_frame.iloc[0]) if len(evacuation_frame) > 0 else None

        # -------------------------
        # (A) 決断前の心理・社会的形成
        # -------------------------
        if decision_frame is None:
            pre = g.copy()
        else:
            pre = g[g["frame"] < decision_frame].copy()

        pre_frames = len(pre)
        pre_mean_density = pre["local_density_before_move"].mean() if pre_frames > 0 else np.nan
        pre_max_density = pre["local_density_before_move"].max() if pre_frames > 0 else np.nan
        pre_dense_frames = (pre["local_density_before_move"] >= dense_threshold).sum() if pre_frames > 0 else 0
        pre_mean_h = pre["h_ext"].mean() if pre_frames > 0 else np.nan
        pre_last_h = pre["h_ext"].iloc[-1] if pre_frames > 0 else np.nan
        pre_mean_V = pre["V_pos"].mean() if pre_frames > 0 else np.nan

        # 決断時の外場
        h_at_decision = decision_frame * cfg.h_step if decision_frame is not None else np.nan

        # 決断までに s=1 に一度もなっていないはずだが、
        # coarse frame での保存なので直前フレーム末に s=1 が見える場合がある。
        pre_s1_fraction = (pre["s"] == 1).mean() if pre_frames > 0 else np.nan

        # -------------------------
        # (B) 決断後の混雑・地形拘束
        # -------------------------
        if decision_frame is None:
            post = g.iloc[0:0].copy()
        else:
            if evacuation_frame is None:
                post = g[g["frame"] >= decision_frame].copy()
            else:
                post = g[(g["frame"] >= decision_frame) & (g["frame"] <= evacuation_frame)].copy()

        post_frames = len(post)
        post_mean_density = post["local_density_before_move"].mean() if post_frames > 0 else np.nan
        post_max_density = post["local_density_before_move"].max() if post_frames > 0 else np.nan
        post_dense_frames = (post["local_density_before_move"] >= dense_threshold).sum() if post_frames > 0 else 0

        post_mean_after_density = post["local_density_after_move"].mean() if post_frames > 0 else np.nan
        post_mean_disp = post["disp_frame"].mean() if post_frames > 0 else np.nan
        post_stagnated_frames = post["stagnated"].sum() if post_frames > 0 else 0
        post_no_progress_frames = post["no_progress"].sum() if post_frames > 0 else 0

        V_at_decision = np.nan
        if decision_frame is not None:
            row_dec = g[g["frame"] == decision_frame]
            if len(row_dec) > 0:
                V_at_decision = row_dec["V_pos"].iloc[0]

        V_at_evac = np.nan
        if evacuation_frame is not None:
            row_evac = g[g["frame"] == evacuation_frame]
            if len(row_evac) > 0:
                V_at_evac = row_evac["V_pos"].iloc[0]

        # -------------------------
        # (C) 空間的滞留の形成
        # -------------------------
        # 同じセルにどれだけ留まったか
        cell_counts = g.groupby(["x", "y"]).size()
        max_residence_same_cell = cell_counts.max() if len(cell_counts) > 0 else 0

        # 決断後だけでの最大滞留
        if len(post) > 0:
            post_cell_counts = post.groupby(["x", "y"]).size()
            post_max_residence_same_cell = post_cell_counts.max()
        else:
            post_max_residence_same_cell = 0

        total_path_length = g["disp_frame"].fillna(0.0).sum()
        net_displacement = np.nan
        if len(g) >= 2:
            dx = g["x"].iloc[-1] - g["x"].iloc[0]
            dy = g["y"].iloc[-1] - g["y"].iloc[0]
            net_displacement = np.sqrt(dx**2 + dy**2)

        records.append({
            "agent_id": agent_id,
            "phi": phi,
            "mu": mu,
            "step_when_decided": step_when_decided,
            "evacuation_time": evacuation_time,
            "decision_frame": decision_frame,
            "evacuation_frame": evacuation_frame,
            "tau_dec_steps": step_when_decided if step_when_decided >= 0 else np.nan,
            "tau_move_steps": (evacuation_time - step_when_decided)
                              if (step_when_decided >= 0 and evacuation_time >= 0)
                              else np.nan,

            # (A) decision-pre
            "pre_frames": pre_frames,
            "pre_mean_density": pre_mean_density,
            "pre_max_density": pre_max_density,
            "pre_dense_frames": pre_dense_frames,
            "pre_mean_h": pre_mean_h,
            "pre_last_h": pre_last_h,
            "h_at_decision": h_at_decision,
            "pre_mean_V": pre_mean_V,
            "pre_s1_fraction": pre_s1_fraction,

            # (B) decision-post
            "post_frames": post_frames,
            "post_mean_density": post_mean_density,
            "post_max_density": post_max_density,
            "post_dense_frames": post_dense_frames,
            "post_mean_after_density": post_mean_after_density,
            "post_mean_disp": post_mean_disp,
            "post_stagnated_frames": post_stagnated_frames,
            "post_no_progress_frames": post_no_progress_frames,
            "V_at_decision": V_at_decision,
            "V_at_evac": V_at_evac,

            # (C) residence
            "max_residence_same_cell": max_residence_same_cell,
            "post_max_residence_same_cell": post_max_residence_same_cell,
            "total_path_length": total_path_length,
            "net_displacement": net_displacement,
        })

    summary = pd.DataFrame(records)
    summary["late_move"] = summary["tau_move_steps"] >= summary["tau_move_steps"].quantile(cfg.late_quantile)
    summary["late_decision"] = summary["tau_dec_steps"] >= summary["tau_dec_steps"].quantile(cfg.late_quantile)
    return summary


# ============================================================
# 4. 可視化
# ============================================================

def plot_pre_decision_formation(summary: pd.DataFrame, outdir: str):
    """
    決断前の心理・社会的形成を見る図。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = summary.dropna(subset=["phi", "tau_dec_steps"])
    sc = ax.scatter(
        valid["phi"], valid["tau_dec_steps"],
        c=valid["pre_mean_density"], alpha=0.6, s=18
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("pre_mean_density")
    ax.set_xlabel("phi")
    ax.set_ylabel(r"$\tau_i^{dec}$ [steps]")
    ax.set_title("Pre-decision psychological/social formation")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "predecision_phi_vs_tau_dec_colored_by_density.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    valid = summary.dropna(subset=["pre_mean_density", "tau_dec_steps"])
    ax.scatter(valid["pre_mean_density"], valid["tau_dec_steps"], alpha=0.6, s=18)
    ax.set_xlabel("pre_mean_density")
    ax.set_ylabel(r"$\tau_i^{dec}$ [steps]")
    ax.set_title("Pre-decision density exposure vs decision delay")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "predecision_density_vs_tau_dec.png"), dpi=200)
    plt.close(fig)


def plot_post_decision_confinement(summary: pd.DataFrame, outdir: str):
    """
    決断後の混雑・地形拘束を見る図。
    """
    valid = summary.dropna(subset=["tau_move_steps"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valid["post_mean_density"], valid["tau_move_steps"], alpha=0.6, s=18)
    ax.set_xlabel("post_mean_density")
    ax.set_ylabel(r"$\tau_i^{move}$ [steps]")
    ax.set_title("Post-decision congestion exposure vs movement delay")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "postdecision_density_vs_tau_move.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valid["post_stagnated_frames"], valid["tau_move_steps"], alpha=0.6, s=18)
    ax.set_xlabel("post_stagnated_frames")
    ax.set_ylabel(r"$\tau_i^{move}$ [steps]")
    ax.set_title("Post-decision stagnation vs movement delay")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "postdecision_stagnation_vs_tau_move.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valid["V_at_decision"], valid["tau_move_steps"], alpha=0.6, s=18)
    ax.set_xlabel("V_at_decision (terrain distance to safety)")
    ax.set_ylabel(r"$\tau_i^{move}$ [steps]")
    ax.set_title("Terrain confinement at decision vs movement delay")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "terrain_V_at_decision_vs_tau_move.png"), dpi=200)
    plt.close(fig)


def _residence_heatmap(panel_subset: pd.DataFrame, grid_w: int, grid_h: int,
                       title: str, outfile: str):
    heat, _, _ = np.histogram2d(
        panel_subset["x"], panel_subset["y"],
        bins=[np.arange(grid_w + 1), np.arange(grid_h + 1)]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        heat.T,
        origin="lower",
        aspect="auto",
        extent=[0, grid_w, 0, grid_h]
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("residence count")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_spatial_residence(panel: pd.DataFrame, summary: pd.DataFrame,
                           grid_w: int, grid_h: int, outdir: str):
    """
    空間的滞留の形成を見る図。
    late_move agent に限定して
      - 全期間 residence
      - 決断前 residence
      - 決断後 residence
    を作る。
    """
    late_ids = set(summary.loc[summary["late_move"], "agent_id"].tolist())
    late_panel = panel[panel["agent_id"].isin(late_ids)].copy()

    if len(late_panel) == 0:
        return

    _residence_heatmap(
        late_panel, grid_w, grid_h,
        "Residence heatmap of late-move agents (all frames)",
        os.path.join(outdir, "residence_heatmap_late_agents_all.png")
    )

    # 決断前
    pre_rows = []
    post_rows = []

    decision_frame_map = summary.set_index("agent_id")["decision_frame"].to_dict()

    for agent_id, g in late_panel.groupby("agent_id"):
        df_dec = decision_frame_map.get(agent_id, np.nan)
        if pd.isna(df_dec):
            pre_rows.append(g)
        else:
            pre_rows.append(g[g["frame"] < df_dec])
            post_rows.append(g[g["frame"] >= df_dec])

    pre_panel = pd.concat(pre_rows, ignore_index=True) if pre_rows else pd.DataFrame(columns=late_panel.columns)
    post_panel = pd.concat(post_rows, ignore_index=True) if post_rows else pd.DataFrame(columns=late_panel.columns)

    if len(pre_panel) > 0:
        _residence_heatmap(
            pre_panel, grid_w, grid_h,
            "Residence heatmap of late-move agents (pre-decision)",
            os.path.join(outdir, "residence_heatmap_late_agents_predecision.png")
        )

    if len(post_panel) > 0:
        _residence_heatmap(
            post_panel, grid_w, grid_h,
            "Residence heatmap of late-move agents (post-decision)",
            os.path.join(outdir, "residence_heatmap_late_agents_postdecision.png")
        )


# ============================================================
# 5. 時系列平均プロファイル
# ============================================================

def make_aligned_profiles(panel: pd.DataFrame, summary: pd.DataFrame, outdir: str, window: int = 30):
    """
    decision_frame を基準に frame を揃えて、
    決断前後の平均密度プロファイルを作る。

    これで「決断のかなり前から混雑曝露が上がっていたのか」を見やすくする。
    """
    decision_map = summary.set_index("agent_id")["decision_frame"].to_dict()

    rows = []
    for agent_id, g in panel.groupby("agent_id"):
        df_dec = decision_map.get(agent_id, np.nan)
        if pd.isna(df_dec):
            continue
        g = g.copy()
        g["rel_frame_from_decision"] = g["frame"] - int(df_dec)
        g = g[(g["rel_frame_from_decision"] >= -window) & (g["rel_frame_from_decision"] <= window)]
        rows.append(g)

    if not rows:
        return

    aligned = pd.concat(rows, ignore_index=True)

    prof = aligned.groupby("rel_frame_from_decision").agg(
        mean_before_density=("local_density_before_move", "mean"),
        mean_after_density=("local_density_after_move", "mean"),
        mean_V=("V_pos", "mean"),
        n=("agent_id", "count")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prof["rel_frame_from_decision"], prof["mean_before_density"], label="before density")
    ax.plot(prof["rel_frame_from_decision"], prof["mean_after_density"], label="after density")
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("frame - decision_frame")
    ax.set_ylabel("mean density")
    ax.set_title("Aligned density profile around decision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "aligned_density_profile_around_decision.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prof["rel_frame_from_decision"], prof["mean_V"])
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("frame - decision_frame")
    ax.set_ylabel("mean V_pos")
    ax.set_title("Aligned terrain-distance profile around decision")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "aligned_V_profile_around_decision.png"), dpi=200)
    plt.close(fig)


# ============================================================
# 6. テーブル保存
# ============================================================

def save_tables(panel: pd.DataFrame, summary: pd.DataFrame, outdir: str):
    """
    panel と agent summary を保存する。
    """
    os.makedirs(outdir, exist_ok=True)
    panel.to_csv(os.path.join(outdir, "timeseries_panel_with_features.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "timeseries_agent_summary.csv"), index=False)

    # 簡単な group summary
    group_summary = pd.DataFrame({
        "n_agents": [len(summary)],
        "n_decided": [summary["tau_dec_steps"].notna().sum()],
        "n_evacuated": [summary["tau_move_steps"].notna().sum()],
        "late_move_count": [summary["late_move"].sum()],
        "late_decision_count": [summary["late_decision"].sum()],
        "mean_tau_dec": [summary["tau_dec_steps"].mean()],
        "median_tau_dec": [summary["tau_dec_steps"].median()],
        "mean_tau_move": [summary["tau_move_steps"].mean()],
        "median_tau_move": [summary["tau_move_steps"].median()],
    })
    group_summary.to_csv(os.path.join(outdir, "timeseries_group_summary.csv"), index=False)


# ============================================================
# 7. 端末サマリ
# ============================================================

def print_summary(summary: pd.DataFrame):
    """
    端末用に主要相関を表示する。
    """
    def corr(a, b):
        tmp = summary[[a, b]].dropna()
        if len(tmp) < 2:
            return np.nan
        return tmp[a].corr(tmp[b])

    print("\n========== Timeseries summary ==========")
    print(f"n_agents = {len(summary)}")
    print(f"decided  = {summary['tau_dec_steps'].notna().sum()}")
    print(f"evacuated= {summary['tau_move_steps'].notna().sum()}")

    print("\n[pre-decision]")
    print("corr(phi, tau_dec_steps)             =", corr("phi", "tau_dec_steps"))
    print("corr(pre_mean_density, tau_dec_steps)=", corr("pre_mean_density", "tau_dec_steps"))
    print("corr(pre_dense_frames, tau_dec_steps)=", corr("pre_dense_frames", "tau_dec_steps"))

    print("\n[post-decision]")
    print("corr(post_mean_density, tau_move_steps)   =", corr("post_mean_density", "tau_move_steps"))
    print("corr(post_stagnated_frames, tau_move_steps)=", corr("post_stagnated_frames", "tau_move_steps"))
    print("corr(post_no_progress_frames, tau_move_steps)=", corr("post_no_progress_frames", "tau_move_steps"))
    print("corr(V_at_decision, tau_move_steps)       =", corr("V_at_decision", "tau_move_steps"))

    print("\n[residence]")
    print("corr(post_max_residence_same_cell, tau_move_steps)=", corr("post_max_residence_same_cell", "tau_move_steps"))
    print("corr(total_path_length, tau_move_steps)           =", corr("total_path_length", "tau_move_steps"))


# ============================================================
# 8. main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="決断前形成・決断後拘束・空間滞留を時系列解析する"
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="agents_frame_*.csv が入っているディレクトリ")
    parser.add_argument("--base_map_csv", type=str, required=True,
                        help="base_map_potential_real_10m.csv のパス")
    parser.add_argument("--output_dir", type=str, default="timeseries_delay_analysis",
                        help="出力先ディレクトリ")
    parser.add_argument("--relaxation_steps", type=int, default=5000)
    parser.add_argument("--h_step", type=float, default=0.005)
    parser.add_argument("--late_quantile", type=float, default=0.90)
    parser.add_argument("--dense_quantile", type=float, default=0.90)

    args = parser.parse_args()
    cfg = Config(
        relaxation_steps=args.relaxation_steps,
        h_step=args.h_step,
        late_quantile=args.late_quantile,
        dense_quantile=args.dense_quantile,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 読み込み
    panel = load_agents_panel(args.results_dir)
    base = load_base_map(args.base_map_csv)

    # 2. 時系列列追加
    panel = add_time_columns(panel, cfg)
    panel = add_motion_columns(panel, base["V_field"])

    # 3. 1 agent ごとの時系列集約
    summary = summarize_per_agent(panel, cfg)

    # 4. 可視化
    plot_pre_decision_formation(summary, args.output_dir)
    plot_post_decision_confinement(summary, args.output_dir)
    plot_spatial_residence(panel, summary, base["grid_w"], base["grid_h"], args.output_dir)
    make_aligned_profiles(panel, summary, args.output_dir, window=30)

    # 5. テーブル保存
    save_tables(panel, summary, args.output_dir)

    # 6. 端末サマリ
    print_summary(summary)

    print("\n[Saved files]")
    for f in sorted(glob.glob(os.path.join(args.output_dir, "*"))):
        print(" -", f)


if __name__ == "__main__":
    main()