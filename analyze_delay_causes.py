import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 基本ユーティリティ
# ============================================================

def extract_frame_number(filepath: str):
    """
    agents_frame_003.csv -> 3
    """
    m = re.search(r"agents_frame_(\d+)\.csv$", os.path.basename(filepath))
    if m is None:
        return None
    return int(m.group(1))


def find_latest_agents_csv(results_dir: str) -> str:
    """
    results_dir から agents_frame_*.csv を探し、最大フレーム番号のものを返す。
    """
    files = sorted(glob.glob(os.path.join(results_dir, "agents_frame_*.csv")))
    if not files:
        raise FileNotFoundError(f"agents_frame_*.csv が見つかりません: {results_dir}")

    frame_file_pairs = []
    for f in files:
        frame = extract_frame_number(f)
        if frame is not None:
            frame_file_pairs.append((frame, f))

    if not frame_file_pairs:
        raise FileNotFoundError("フレーム番号を抽出できる agents_frame_*.csv がありません。")

    frame_file_pairs.sort(key=lambda x: x[0])
    return frame_file_pairs[-1][1]


def load_grid_shape_from_base_map(base_map_csv: str):
    """
    base_map_potential_real_10m.csv から GRID_W, GRID_H を取得する。
    シミュレーション本体と同じく、
        GRID_W = len(unique(lon))
        GRID_H = len(unique(lat))
    とする。
    """
    if not os.path.exists(base_map_csv):
        raise FileNotFoundError(f"base map CSV が見つかりません: {base_map_csv}")

    df_map = pd.read_csv(base_map_csv)
    if "lon" not in df_map.columns or "lat" not in df_map.columns:
        raise ValueError("base_map_csv に lon, lat 列が必要です。")

    grid_w = len(np.sort(df_map["lon"].unique()))
    grid_h = len(np.sort(df_map["lat"].unique()))
    return grid_w, grid_h


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson"):
    """
    欠損を落として相関を返す。
    """
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 2:
        return np.nan
    return tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method=method)


# ============================================================
# 2. 解析に使う派生量を作る
# ============================================================

def build_analysis_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    最終フレームの DataFrame から、解析用の列を追加する。

    追加する主な列:
      tau_dec  = step_when_decided
      tau_move = evacuation_time - step_when_decided
      decided
      evacuated
      delta_density = after - before
    """
    required = [
        "agent_id", "phi", "mu",
        "step_when_decided", "evacuation_time",
        "local_density_before_move", "local_density_after_move",
        "x", "y", "rank", "is_safe"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"必要列が不足しています: {missing}")

    out = df.copy()

    out["decided"] = out["step_when_decided"] != -1
    out["evacuated"] = out["evacuation_time"] != -1

    # 決断遅れ τ_dec
    out["tau_dec"] = np.where(out["decided"], out["step_when_decided"], np.nan)

    # 移動遅れ τ_move = evacuation_time - step_when_decided
    out["tau_move"] = np.where(
        out["decided"] & out["evacuated"],
        out["evacuation_time"] - out["step_when_decided"],
        np.nan
    )

    # 密度差
    out["delta_density"] = (
        out["local_density_after_move"] - out["local_density_before_move"]
    )

    return out


# ============================================================
# 3. 分類ロジック
# ============================================================

def classify_delay_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    決断遅れ vs 移動遅れ の散布図に基づく粗い分類を行う。

    分類の考え方:
      - tau_dec の中央値より大きい -> 決断遅れ側
      - tau_move の中央値より大きい -> 移動遅れ側

    これで
      心理遅れ型 / 渋滞遅れ型 / 混合型 / 早期型
    に分ける。
    """
    out = df.copy()

    valid = out.dropna(subset=["tau_dec", "tau_move"]).copy()
    if len(valid) == 0:
        out["delay_type"] = "undefined"
        return out

    dec_thr = valid["tau_dec"].median()
    move_thr = valid["tau_move"].median()

    def classify_row(row):
        if pd.isna(row["tau_dec"]) or pd.isna(row["tau_move"]):
            if not row["decided"]:
                return "undecided"
            if row["decided"] and not row["evacuated"]:
                return "decided_not_evacuated"
            return "undefined"

        dec_late = row["tau_dec"] > dec_thr
        move_late = row["tau_move"] > move_thr

        if dec_late and not move_late:
            return "psychological_delay"
        elif (not dec_late) and move_late:
            return "congestion_delay"
        elif dec_late and move_late:
            return "mixed_delay"
        else:
            return "early_or_small_delay"

    out["delay_type"] = out.apply(classify_row, axis=1)
    out.attrs["tau_dec_threshold"] = dec_thr
    out.attrs["tau_move_threshold"] = move_thr
    return out


# ============================================================
# 4. 図1: 決断遅れ vs 移動遅れ
# ============================================================

def plot_decision_vs_move_delay(df: pd.DataFrame, output_path: str):
    """
    横軸 tau_dec, 縦軸 tau_move の散布図を描く。
    中央値で区切り線を入れて、4象限の解釈をしやすくする。
    """
    plot_df = df.dropna(subset=["tau_dec", "tau_move"]).copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(plot_df["tau_dec"], plot_df["tau_move"], alpha=0.5, s=15)

    dec_thr = plot_df["tau_dec"].median()
    move_thr = plot_df["tau_move"].median()

    ax.axvline(dec_thr, linestyle="--")
    ax.axhline(move_thr, linestyle="--")

    ax.set_xlabel(r"$\tau_i^{dec} = \mathrm{step\_when\_decided}$")
    ax.set_ylabel(r"$\tau_i^{move} = \mathrm{evacuation\_time} - \mathrm{step\_when\_decided}$")
    ax.set_title("Decision delay vs movement delay")

    # 象限ラベル
    ax.text(dec_thr * 1.02, move_thr * 0.3 if move_thr > 0 else 0.1,
            "psychological\n delay", fontsize=10)
    ax.text(dec_thr * 0.3 if dec_thr > 0 else 0.1, move_thr * 1.02,
            "congestion\n delay", fontsize=10)
    ax.text(dec_thr * 1.02, move_thr * 1.02,
            "mixed\n delay", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================
# 5. 図2: phi と決断時刻
# ============================================================

def plot_phi_vs_tau_dec(df: pd.DataFrame, output_path: str):
    """
    phi と tau_dec の散布図を描き、Pearson/Spearman 相関を表示する。
    """
    plot_df = df.dropna(subset=["tau_dec"]).copy()

    pearson = safe_corr(plot_df["phi"], plot_df["tau_dec"], method="pearson")
    spearman = safe_corr(plot_df["phi"], plot_df["tau_dec"], method="spearman")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_df["phi"], plot_df["tau_dec"], alpha=0.5, s=15)

    if len(plot_df) >= 2:
        # 一次回帰線
        coef = np.polyfit(plot_df["phi"], plot_df["tau_dec"], deg=1)
        x_line = np.linspace(plot_df["phi"].min(), plot_df["phi"].max(), 200)
        y_line = coef[0] * x_line + coef[1]
        ax.plot(x_line, y_line)

    ax.set_xlabel("phi")
    ax.set_ylabel(r"$\tau_i^{dec}$")
    ax.set_title(
        f"phi vs decision delay\nPearson={pearson:.3f}, Spearman={spearman:.3f}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================
# 6. 図3: before/after 密度 と tau_move
# ============================================================

def plot_density_vs_tau_move(df: pd.DataFrame, output_dir: str):
    """
    before density と tau_move
    after density と tau_move
    の2つの散布図を作る。
    """
    plot_df = df.dropna(subset=["tau_move"]).copy()

    # before
    pearson_before = safe_corr(
        plot_df["local_density_before_move"], plot_df["tau_move"], method="pearson"
    )
    spearman_before = safe_corr(
        plot_df["local_density_before_move"], plot_df["tau_move"], method="spearman"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_df["local_density_before_move"], plot_df["tau_move"], alpha=0.5, s=15)
    ax.set_xlabel("local_density_before_move")
    ax.set_ylabel(r"$\tau_i^{move}$")
    ax.set_title(
        f"Before density vs movement delay\nPearson={pearson_before:.3f}, Spearman={spearman_before:.3f}"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "density_before_vs_tau_move.png"), dpi=200)
    plt.close(fig)

    # after
    pearson_after = safe_corr(
        plot_df["local_density_after_move"], plot_df["tau_move"], method="pearson"
    )
    spearman_after = safe_corr(
        plot_df["local_density_after_move"], plot_df["tau_move"], method="spearman"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_df["local_density_after_move"], plot_df["tau_move"], alpha=0.5, s=15)
    ax.set_xlabel("local_density_after_move")
    ax.set_ylabel(r"$\tau_i^{move}$")
    ax.set_title(
        f"After density vs movement delay\nPearson={pearson_after:.3f}, Spearman={spearman_after:.3f}"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "density_after_vs_tau_move.png"), dpi=200)
    plt.close(fig)


# ============================================================
# 7. 図4: 遅い人の x,y ヒートマップ
# ============================================================

def plot_slow_agents_heatmap(df: pd.DataFrame, grid_w: int, grid_h: int,
                             output_path: str, use_metric: str = "tau_move",
                             top_quantile: float = 0.9):
    """
    遅い人の最終位置 (x,y) の 2D ヒートマップを描く。

    use_metric:
      - "tau_move": 決断後の移動遅れが大きい人
      - "evacuation_time": 避難完了時刻が大きい人
    """
    plot_df = df.copy()

    if use_metric not in ["tau_move", "evacuation_time"]:
        raise ValueError("use_metric は 'tau_move' または 'evacuation_time' を指定してください。")

    # 未避難者も見たいので、evacuation_time を使う場合は -1 を除く
    plot_df = plot_df.dropna(subset=[use_metric])
    plot_df = plot_df[plot_df[use_metric] != -1]

    if len(plot_df) == 0:
        raise ValueError("ヒートマップ用の有効データがありません。")

    thr = plot_df[use_metric].quantile(top_quantile)
    slow_df = plot_df[plot_df[use_metric] >= thr].copy()

    heat, xedges, yedges = np.histogram2d(
        slow_df["x"], slow_df["y"],
        bins=[np.arange(grid_w + 1), np.arange(grid_h + 1)]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        heat.T,
        origin="lower",
        aspect="auto",
        extent=[0, grid_w, 0, grid_h]
    )
    fig.colorbar(im, ax=ax, label="count of slow agents")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Heatmap of slow agents ({use_metric}, top {int((1-top_quantile)*100)}%)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================
# 8. 図5: rank 別の遅れ分布
# ============================================================

def plot_rank_delay_distribution(df: pd.DataFrame, output_dir: str):
    """
    rank ごとの tau_dec, tau_move の分布を箱ひげ図で見る。
    数値実装由来の偏り確認に使う。
    """
    # tau_dec
    plot_df_dec = df.dropna(subset=["tau_dec"]).copy()
    ranks_dec = sorted(plot_df_dec["rank"].unique())
    data_dec = [plot_df_dec.loc[plot_df_dec["rank"] == r, "tau_dec"].values for r in ranks_dec]

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(data_dec) > 0:
        ax.boxplot(data_dec, labels=ranks_dec, showfliers=False)
    ax.set_xlabel("rank")
    ax.set_ylabel(r"$\tau_i^{dec}$")
    ax.set_title("Decision delay distribution by rank")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rank_vs_tau_dec_boxplot.png"), dpi=200)
    plt.close(fig)

    # tau_move
    plot_df_move = df.dropna(subset=["tau_move"]).copy()
    ranks_move = sorted(plot_df_move["rank"].unique())
    data_move = [plot_df_move.loc[plot_df_move["rank"] == r, "tau_move"].values for r in ranks_move]

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(data_move) > 0:
        ax.boxplot(data_move, labels=ranks_move, showfliers=False)
    ax.set_xlabel("rank")
    ax.set_ylabel(r"$\tau_i^{move}$")
    ax.set_title("Movement delay distribution by rank")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rank_vs_tau_move_boxplot.png"), dpi=200)
    plt.close(fig)


# ============================================================
# 9. 数値サマリ
# ============================================================

def save_summary_tables(df: pd.DataFrame, output_dir: str):
    """
    解析用の集計表を CSV として保存する。
    """
    # 分類集計
    type_counts = df["delay_type"].value_counts(dropna=False).rename_axis("delay_type").reset_index(name="count")
    type_counts.to_csv(os.path.join(output_dir, "delay_type_counts.csv"), index=False)

    # rank 集計
    rank_summary = df.groupby("rank").agg(
        n_agents=("agent_id", "count"),
        mean_tau_dec=("tau_dec", "mean"),
        median_tau_dec=("tau_dec", "median"),
        mean_tau_move=("tau_move", "mean"),
        median_tau_move=("tau_move", "median"),
        mean_phi=("phi", "mean"),
        mean_before_density=("local_density_before_move", "mean"),
        mean_after_density=("local_density_after_move", "mean"),
    ).reset_index()
    rank_summary.to_csv(os.path.join(output_dir, "rank_delay_summary.csv"), index=False)

    # 個体ごとの解析テーブル
    analysis_cols = [
        "agent_id", "x", "y", "rank",
        "phi", "mu",
        "step_when_decided", "evacuation_time",
        "tau_dec", "tau_move",
        "local_density_before_move", "local_density_after_move",
        "delta_density",
        "delay_type", "is_safe"
    ]
    df[analysis_cols].to_csv(os.path.join(output_dir, "agent_delay_analysis.csv"), index=False)


# ============================================================
# 10. メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="避難遅れ要因の解析スクリプト")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="agents_frame_*.csv があるディレクトリ")
    parser.add_argument("--base_map_csv", type=str, required=True,
                        help="base_map_potential_real_10m.csv のパス")
    parser.add_argument("--agents_csv", type=str, default=None,
                        help="解析対象の agents_frame_*.csv を直接指定したい場合")
    parser.add_argument("--output_dir", type=str, default="delay_analysis_results",
                        help="解析結果の保存先")
    parser.add_argument("--slow_metric", type=str, default="tau_move",
                        choices=["tau_move", "evacuation_time"],
                        help="ヒートマップで遅い人を定義する指標")
    parser.add_argument("--slow_quantile", type=float, default=0.9,
                        help="遅い人の上位分位点。0.9 なら上位10%")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. base map から格子サイズ取得
    grid_w, grid_h = load_grid_shape_from_base_map(args.base_map_csv)

    # 2. 解析対象CSVを決定
    if args.agents_csv is None:
        agents_csv = find_latest_agents_csv(args.results_dir)
    else:
        agents_csv = args.agents_csv

    print(f"[INFO] use agents csv : {agents_csv}")
    print(f"[INFO] GRID_W={grid_w}, GRID_H={grid_h}")

    # 3. 読み込み
    df = pd.read_csv(agents_csv)

    # 4. 解析用列を作る
    df = build_analysis_dataframe(df)

    # 5. 遅れ型分類
    df = classify_delay_types(df)

    # 6. 図の出力
    plot_decision_vs_move_delay(df, os.path.join(args.output_dir, "decision_vs_move_delay.png"))
    plot_phi_vs_tau_dec(df, os.path.join(args.output_dir, "phi_vs_tau_dec.png"))
    plot_density_vs_tau_move(df, args.output_dir)
    plot_slow_agents_heatmap(
        df,
        grid_w=grid_w,
        grid_h=grid_h,
        output_path=os.path.join(args.output_dir, "slow_agents_heatmap.png"),
        use_metric=args.slow_metric,
        top_quantile=args.slow_quantile
    )
    plot_rank_delay_distribution(df, args.output_dir)

    # 7. 集計表
    save_summary_tables(df, args.output_dir)

    # 8. 端末表示用サマリ
    print("\n========== Summary ==========")
    print(f"n_agents = {len(df)}")
    print(f"decided  = {(df['decided']).sum()}")
    print(f"evacuated= {(df['evacuated']).sum()}")

    print("\n[delay_type counts]")
    print(df["delay_type"].value_counts(dropna=False))

    print("\n[correlations]")
    print("phi vs tau_dec (pearson)      =", safe_corr(df["phi"], df["tau_dec"], method="pearson"))
    print("phi vs tau_dec (spearman)     =", safe_corr(df["phi"], df["tau_dec"], method="spearman"))
    print("before_density vs tau_move    =", safe_corr(df["local_density_before_move"], df["tau_move"], method="pearson"))
    print("after_density vs tau_move     =", safe_corr(df["local_density_after_move"], df["tau_move"], method="pearson"))

    print("\n[Saved files]")
    for f in sorted(glob.glob(os.path.join(args.output_dir, "*"))):
        print(" -", f)


if __name__ == "__main__":
    main()

    