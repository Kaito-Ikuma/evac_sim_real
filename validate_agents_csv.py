import os
import re
import glob
import argparse
import numpy as np
import pandas as pd


# ============================================================
# 1. 補助関数
# ============================================================

def is_integer_like(series, tol=1e-9):
    """
    浮動小数で保存されていても、実質的に整数なら True とみなす。
    """
    return np.isclose(series, np.round(series), atol=tol)


def extract_frame_number(filepath):
    """
    ファイル名 agents_frame_003.csv からフレーム番号 3 を取り出す。
    """
    m = re.search(r"agents_frame_(\d+)\.csv$", os.path.basename(filepath))
    if m is None:
        return None
    return int(m.group(1))


def expected_max_step(frame, relaxation_steps):
    """
    frame が終わった時点で到達可能な最大 step index
    """
    return (frame + 1) * relaxation_steps - 1


def print_check_result(ok, label, detail=""):
    """
    チェック結果を見やすく表示する。
    """
    status = "[OK]" if ok else "[NG]"
    if detail:
        print(f"{status} {label}: {detail}")
    else:
        print(f"{status} {label}")


def load_grid_shape_from_base_map(base_map_csv):
    """
    base_map_potential_real_10m.csv から GRID_W, GRID_H を自動取得する。

    シミュレーション本体と同様に
        GRID_W = len(unique(lon))
        GRID_H = len(unique(lat))
    と定義する。
    """
    if not os.path.exists(base_map_csv):
        raise FileNotFoundError(f"base map CSV が見つかりません: {base_map_csv}")

    df_map = pd.read_csv(base_map_csv)

    required_cols = ["lon", "lat"]
    missing = [c for c in required_cols if c not in df_map.columns]
    if missing:
        raise ValueError(f"base map CSV に必要な列がありません: {missing}")

    lon_unique = np.sort(df_map["lon"].unique())
    lat_unique = np.sort(df_map["lat"].unique())

    grid_w = len(lon_unique)
    grid_h = len(lat_unique)

    return grid_w, grid_h


# ============================================================
# 2. 1フレーム単位の検証
# ============================================================

def validate_single_frame(df, frame, n_agents, grid_w, grid_h, relaxation_steps):
    errors = []
    warnings = []

    required_columns = [
        "agent_id", "x", "y", "s", "phi", "mu",
        "local_density_before_move", "local_density_after_move",
        "step_when_decided", "evacuation_time",
        "rank", "is_safe"
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        errors.append(f"必須列が不足: {missing}")
        return errors, warnings

    if len(df) != n_agents:
        errors.append(f"行数が N_AGENTS と一致しない: rows={len(df)}, expected={n_agents}")

    duplicated_ids = df["agent_id"][df["agent_id"].duplicated()]
    if not duplicated_ids.empty:
        errors.append(f"agent_id に重複あり: {duplicated_ids.tolist()[:10]} ...")

    expected_ids = set(range(n_agents))
    actual_ids = set(df["agent_id"].astype(int).tolist())
    missing_ids = expected_ids - actual_ids
    extra_ids = actual_ids - expected_ids
    if missing_ids:
        errors.append(f"agent_id に欠番あり（例）: {sorted(list(missing_ids))[:10]} ...")
    if extra_ids:
        errors.append(f"想定外の agent_id あり（例）: {sorted(list(extra_ids))[:10]} ...")

    bad_x = df[(df["x"] < 0) | (df["x"] >= grid_w)]
    bad_y = df[(df["y"] < 0) | (df["y"] >= grid_h)]
    if not bad_x.empty:
        errors.append(f"x が範囲外の行あり（例）:\n{bad_x[['agent_id','x']].head()}")
    if not bad_y.empty:
        errors.append(f"y が範囲外の行あり（例）:\n{bad_y[['agent_id','y']].head()}")

    bad_s = df[~df["s"].isin([-1, 1])]
    if not bad_s.empty:
        errors.append(f"s が -1,1 以外の値を持つ行あり（例）:\n{bad_s[['agent_id','s']].head()}")

    bad_before_neg = df[df["local_density_before_move"] < 0]
    bad_after_neg = df[df["local_density_after_move"] < 0]
    if not bad_before_neg.empty:
        errors.append(f"local_density_before_move に負値あり（例）:\n{bad_before_neg[['agent_id','local_density_before_move']].head()}")
    if not bad_after_neg.empty:
        errors.append(f"local_density_after_move に負値あり（例）:\n{bad_after_neg[['agent_id','local_density_after_move']].head()}")

    bad_before_nonint = df[~is_integer_like(df["local_density_before_move"])]
    bad_after_nonint = df[~is_integer_like(df["local_density_after_move"])]
    if not bad_before_nonint.empty:
        warnings.append(f"local_density_before_move に整数でない値あり（例）:\n{bad_before_nonint[['agent_id','local_density_before_move']].head()}")
    if not bad_after_nonint.empty:
        warnings.append(f"local_density_after_move に整数でない値あり（例）:\n{bad_after_nonint[['agent_id','local_density_after_move']].head()}")

    bad_before_large = df[df["local_density_before_move"] > n_agents]
    bad_after_large = df[df["local_density_after_move"] > n_agents]
    if not bad_before_large.empty:
        errors.append(f"local_density_before_move > N_AGENTS の行あり（例）:\n{bad_before_large[['agent_id','local_density_before_move']].head()}")
    if not bad_after_large.empty:
        errors.append(f"local_density_after_move > N_AGENTS の行あり（例）:\n{bad_after_large[['agent_id','local_density_after_move']].head()}")

    max_step = expected_max_step(frame, relaxation_steps)

    bad_decision_low = df[(df["step_when_decided"] < -1)]
    bad_decision_high = df[(df["step_when_decided"] > max_step)]
    if not bad_decision_low.empty:
        errors.append(f"step_when_decided < -1 の行あり（例）:\n{bad_decision_low[['agent_id','step_when_decided']].head()}")
    if not bad_decision_high.empty:
        errors.append(f"step_when_decided が未来時刻（>{max_step}）の行あり（例）:\n{bad_decision_high[['agent_id','step_when_decided']].head()}")

    bad_evac_low = df[(df["evacuation_time"] < -1)]
    bad_evac_high = df[(df["evacuation_time"] > max_step)]
    if not bad_evac_low.empty:
        errors.append(f"evacuation_time < -1 の行あり（例）:\n{bad_evac_low[['agent_id','evacuation_time']].head()}")
    if not bad_evac_high.empty:
        errors.append(f"evacuation_time が未来時刻（>{max_step}）の行あり（例）:\n{bad_evac_high[['agent_id','evacuation_time']].head()}")

    bad_order = df[
        (df["evacuation_time"] != -1) &
        (df["step_when_decided"] != -1) &
        (df["evacuation_time"] < df["step_when_decided"])
    ]
    if not bad_order.empty:
        errors.append(f"evacuation_time < step_when_decided の行あり（例）:\n{bad_order[['agent_id','step_when_decided','evacuation_time']].head()}")

    bad_safe = df[(df["evacuation_time"] != -1) & (df["is_safe"] != 1)]
    if not bad_safe.empty:
        errors.append(f"evacuation_time != -1 なのに is_safe != 1 の行あり（例）:\n{bad_safe[['agent_id','evacuation_time','is_safe']].head()}")

    return errors, warnings


# ============================================================
# 3. フレームをまたいだ検証
# ============================================================

def validate_across_frames(frames_data, n_agents):
    errors = []
    warnings = []

    frames_sorted = sorted(frames_data.keys())

    expected_ids = set(range(n_agents))
    for frame in frames_sorted:
        df = frames_data[frame]
        actual_ids = set(df["agent_id"].astype(int).tolist())
        if actual_ids != expected_ids:
            missing_ids = expected_ids - actual_ids
            extra_ids = actual_ids - expected_ids
            if missing_ids:
                errors.append(f"[frame={frame}] 欠番あり（例）: {sorted(list(missing_ids))[:10]} ...")
            if extra_ids:
                errors.append(f"[frame={frame}] 想定外 ID あり（例）: {sorted(list(extra_ids))[:10]} ...")

    df_all = []
    for frame in frames_sorted:
        df = frames_data[frame].copy()
        df["frame"] = frame
        df_all.append(df)
    df_all = pd.concat(df_all, ignore_index=True)
    df_all = df_all.sort_values(["agent_id", "frame"]).reset_index(drop=True)

    for agent_id, g in df_all.groupby("agent_id"):
        vals = g["step_when_decided"].tolist()
        fixed_value = None
        for v in vals:
            if v != -1:
                if fixed_value is None:
                    fixed_value = v
                elif v != fixed_value:
                    errors.append(f"agent_id={agent_id}: step_when_decided が途中で変化 (例: {vals})")
                    break

    for agent_id, g in df_all.groupby("agent_id"):
        vals = g["evacuation_time"].tolist()
        fixed_value = None
        for v in vals:
            if v != -1:
                if fixed_value is None:
                    fixed_value = v
                elif v != fixed_value:
                    errors.append(f"agent_id={agent_id}: evacuation_time が途中で変化 (例: {vals})")
                    break

    for agent_id, g in df_all.groupby("agent_id"):
        safe_vals = g["is_safe"].tolist()
        seen_safe = False
        for v in safe_vals:
            if v == 1:
                seen_safe = True
            if seen_safe and v != 1:
                errors.append(f"agent_id={agent_id}: 一度 is_safe=1 になった後に 0 に戻っている (例: {safe_vals})")
                break

    return errors, warnings, df_all


# ============================================================
# 4. サマリ表示
# ============================================================

def print_frame_summary(frame, df):
    n_decided = (df["step_when_decided"] != -1).sum()
    n_evacuated = (df["evacuation_time"] != -1).sum()

    delta_rho = df["local_density_after_move"] - df["local_density_before_move"]

    print(f"\n--- Frame {frame:03d} summary ---")
    print(f"rows: {len(df)}")
    print(f"decided count   : {n_decided}")
    print(f"evacuated count : {n_evacuated}")

    print("\nlocal_density_before_move summary")
    print(df["local_density_before_move"].describe())

    print("\nlocal_density_after_move summary")
    print(df["local_density_after_move"].describe())

    print("\ndelta_rho = after - before summary")
    print(delta_rho.describe())


# ============================================================
# 5. メイン処理
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="agents_frame_*.csv を検証するスクリプト"
    )
    parser.add_argument(
        "--dir", type=str, required=True,
        help="agents_frame_*.csv が保存されているディレクトリ"
    )
    parser.add_argument(
        "--n_agents", type=int, default=2000,
        help="期待するエージェント数"
    )
    parser.add_argument(
        "--grid_w", type=int, default=None,
        help="格子の x 方向サイズ GRID_W。未指定なら --base_map_csv から自動取得"
    )
    parser.add_argument(
        "--grid_h", type=int, default=None,
        help="格子の y 方向サイズ GRID_H。未指定なら --base_map_csv から自動取得"
    )
    parser.add_argument(
        "--base_map_csv", type=str, default=None,
        help="base_map_potential_real_10m.csv のパス。GRID_W, GRID_H 自動取得に使う"
    )
    parser.add_argument(
        "--relaxation_steps", type=int, default=5000,
        help="1フレームあたりの RELAXATION_STEPS"
    )
    parser.add_argument(
        "--show_summary", action="store_true",
        help="各フレームの密度統計も表示する"
    )

    args = parser.parse_args()

    # ============================================================
    # GRID_W, GRID_H の決定
    # ============================================================
    if args.grid_w is not None and args.grid_h is not None:
        grid_w = args.grid_w
        grid_h = args.grid_h
        print(f"[INFO] 手動指定の GRID_W, GRID_H を使用: GRID_W={grid_w}, GRID_H={grid_h}")
    else:
        if args.base_map_csv is None:
            raise ValueError(
                "GRID_W, GRID_H を自動取得するには --base_map_csv を指定してください。"
            )
        grid_w, grid_h = load_grid_shape_from_base_map(args.base_map_csv)
        print(f"[INFO] base_map_csv から自動取得: GRID_W={grid_w}, GRID_H={grid_h}")

    pattern = os.path.join(args.dir, "agents_frame_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[NG] {pattern} に一致するファイルが見つかりません。")
        return

    frames_data = {}
    all_errors = []
    all_warnings = []

    print("============================================================")
    print("各フレームの検証")
    print("============================================================")

    for filepath in files:
        frame = extract_frame_number(filepath)
        if frame is None:
            all_warnings.append(f"フレーム番号を抽出できないファイルをスキップ: {filepath}")
            continue

        df = pd.read_csv(filepath)
        frames_data[frame] = df

        errors, warnings = validate_single_frame(
            df=df,
            frame=frame,
            n_agents=args.n_agents,
            grid_w=grid_w,
            grid_h=grid_h,
            relaxation_steps=args.relaxation_steps
        )

        ok = (len(errors) == 0)
        print_check_result(ok, f"frame={frame:03d}")

        if errors:
            for e in errors:
                print("   ERROR:", e)
                all_errors.append(f"[frame={frame:03d}] {e}")
        if warnings:
            for w in warnings:
                print("   WARN :", w)
                all_warnings.append(f"[frame={frame:03d}] {w}")

        if args.show_summary:
            print_frame_summary(frame, df)

    print("\n============================================================")
    print("フレームをまたいだ検証")
    print("============================================================")

    cross_errors, cross_warnings, df_all = validate_across_frames(
        frames_data=frames_data,
        n_agents=args.n_agents
    )

    print_check_result(len(cross_errors) == 0, "frame-cross checks")

    if cross_errors:
        for e in cross_errors:
            print("   ERROR:", e)
            all_errors.append(e)

    if cross_warnings:
        for w in cross_warnings:
            print("   WARN :", w)
            all_warnings.append(w)

    print("\n============================================================")
    print("研究向けサマリ")
    print("============================================================")

    last_frame = max(frames_data.keys())
    df_last = frames_data[last_frame].copy()

    df_last["move_delay"] = np.where(
        (df_last["evacuation_time"] != -1) & (df_last["step_when_decided"] != -1),
        df_last["evacuation_time"] - df_last["step_when_decided"],
        np.nan
    )

    print(f"最終フレーム = {last_frame:03d}")
    print(f"決断済み人数   = {(df_last['step_when_decided'] != -1).sum()}")
    print(f"避難完了人数   = {(df_last['evacuation_time'] != -1).sum()}")

    print("\nmove_delay = evacuation_time - step_when_decided の統計")
    print(df_last["move_delay"].describe())

    print("\nlocal_density_before_move と move_delay の相関")
    print(df_last[["local_density_before_move", "move_delay"]].corr())

    print("\nlocal_density_after_move と move_delay の相関")
    print(df_last[["local_density_after_move", "move_delay"]].corr())

    print("\n============================================================")
    print("最終判定")
    print("============================================================")
    print(f"ERROR 数   = {len(all_errors)}")
    print(f"WARNING 数 = {len(all_warnings)}")

    if all_errors:
        print("\n---- ERROR 一覧 ----")
        for e in all_errors:
            print(e)

    if all_warnings:
        print("\n---- WARNING 一覧 ----")
        for w in all_warnings:
            print(w)

    if len(all_errors) == 0:
        print("\n[OK] 大きな矛盾は見つかりませんでした。")
    else:
        print("\n[NG] 上記のエラーを確認してください。")


if __name__ == "__main__":
    main()