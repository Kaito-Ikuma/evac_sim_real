import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import box

try:
    import geopandas as gpd
    from sqlalchemy import create_engine
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

import skfmm


# ============================================================
# 0. 設定
# ============================================================

BASE_MAP_CSV = "base_map_potential_real_10m.csv"
MACRO_STATS_CSV = "simulation_results_threshold_10m/simulation_results_threshold_10m/macro_stats.csv"  # 必要に応じて変更
OUTPUT_DIR = "figures_for_slides"

# study_area_map 用（Method B）
DB_URL = "postgresql://postgres:mysecretpassword@localhost:5432/evacuation"
USE_DB_FOR_STUDY_AREA = True

# 対象領域（既存コードと同じ）
MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = 138.22, 36.65, 138.30, 36.74

# study area map の英語タイトル
STUDY_AREA_TITLE = "Study area: Naganuma district, Chikuma River basin"

# 図の保存 DPI
DPI = 300

# 色設定
COLOR_OBSTACLE = "#000000"      # 黒
COLOR_PASSABLE = "#FFFFFF"      # 白
COLOR_SAFE = "#d9f2d9"          # 薄い緑
COLOR_HAZARD = "#fff2cc"        # 淡い黄（mesh overview で使用）


# ============================================================
# 1. 共通データ読み込み
# ============================================================

def load_base_map_csv(csv_path):
    """
    base_map_potential_real_10m.csv を読み込み、
    シミュレーションと同じ 2D グリッドへ戻す。
    """
    df = pd.read_csv(csv_path)

    required_cols = ["mesh_id", "lon", "lat", "potential_v", "is_obstacle"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV に必要な列がありません: {missing}")

    # bool に正規化
    if df["is_obstacle"].dtype != bool:
        df["is_obstacle"] = df["is_obstacle"].astype(bool)

    # pivot で 2D 配列へ
    pot = (
        df.pivot(index="lat", columns="lon", values="potential_v")
          .sort_index(axis=0)
          .sort_index(axis=1)
    )
    obs = (
        df.pivot(index="lat", columns="lon", values="is_obstacle")
          .sort_index(axis=0)
          .sort_index(axis=1)
    )

    lats = pot.index.to_numpy()
    lons = pot.columns.to_numpy()

    potential_v = pot.to_numpy()
    obstacle_mask = obs.to_numpy().astype(bool)

    # 安全圏: potential_v == 0 かつ obstacle ではない
    safe_mask = (potential_v == 0) & (~obstacle_mask)

    # 危険域（通行可能）: potential_v > 0 かつ obstacle ではない
    hazard_passable_mask = (potential_v > 0) & (~obstacle_mask)

    # extent は imshow 用
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    return {
        "df": df,
        "lons": lons,
        "lats": lats,
        "potential_v": potential_v,
        "obstacle_mask": obstacle_mask,
        "safe_mask": safe_mask,
        "hazard_passable_mask": hazard_passable_mask,
        "extent": extent
    }


def compute_distance_field(potential_v, obstacle_mask):
    """
    シミュレーション実装と同じ考え方で、
    safe zone (potential_v == 0) からの FMM 距離場を計算する。
    """
    phi = np.ones_like(potential_v, dtype=float)
    phi[potential_v == 0] = 0.0

    phi_masked = np.ma.MaskedArray(phi, mask=obstacle_mask)
    distance_field = skfmm.distance(phi_masked).data
    distance_field[obstacle_mask] = np.inf

    return distance_field


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. 図1: study_area_map.png
#    Method B: GeoPandas / hazard_map を優先
#    失敗したら CSV ベースへフォールバック
# ============================================================

def plot_study_area_map(output_path, db_url, use_db, base_data):
    """
    英語タイトルで study_area_map.png を作る。
    優先:
      1) PostGIS hazard_map を GeoPandas で読む
      2) 失敗したら CSV のマスクから簡易表示
    """
    if use_db and HAS_GPD:
        try:
            engine = create_engine(db_url)
            hazard_gdf = gpd.read_postgis(
                "SELECT * FROM hazard_map;",
                con=engine,
                geom_col="geometry"
            )

            if hazard_gdf.crs is None:
                hazard_gdf = hazard_gdf.set_crs("EPSG:4326")
            else:
                hazard_gdf = hazard_gdf.to_crs("EPSG:4326")

            fig, ax = plt.subplots(figsize=(8, 8))

            # ハザード領域
            hazard_gdf.plot(
                ax=ax,
                color="#f4a6a6",
                edgecolor="#b55d5d",
                linewidth=0.4,
                alpha=0.85
            )

            # 対象領域枠
            bbox_rect = Rectangle(
                (MIN_LON, MIN_LAT),
                MAX_LON - MIN_LON,
                MAX_LAT - MIN_LAT,
                fill=False,
                edgecolor="black",
                linewidth=2.0,
                linestyle="--"
            )
            ax.add_patch(bbox_rect)

            ax.set_xlim(MIN_LON, MAX_LON)
            ax.set_ylim(MIN_LAT, MAX_LAT)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(STUDY_AREA_TITLE, fontsize=14)

            legend_handles = [
                Patch(facecolor="#f4a6a6", edgecolor="#b55d5d", label="Hazard area"),
                Patch(facecolor="none", edgecolor="black", label="Simulation domain")
            ]
            ax.legend(handles=legend_handles, loc="upper right", frameon=True)

            ax.grid(True, linestyle=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)

            print(f"[OK] {output_path} を保存しました（PostGIS / hazard_map 使用）")
            return

        except Exception as e:
            print(f"[WARN] PostGIS からの study_area_map 作成に失敗しました。CSV 表示へフォールバックします。詳細: {e}")

    # ---------- フォールバック ----------
    potential_v = base_data["potential_v"]
    obstacle_mask = base_data["obstacle_mask"]
    safe_mask = base_data["safe_mask"]
    hazard_passable_mask = base_data["hazard_passable_mask"]
    extent = base_data["extent"]

    # 0: obstacle, 1: safe, 2: hazard-passable
    category = np.full_like(potential_v, fill_value=np.nan, dtype=float)
    category[hazard_passable_mask] = 2
    category[safe_mask] = 1
    category[obstacle_mask] = 0

    cmap = ListedColormap([COLOR_OBSTACLE, COLOR_SAFE, "#f4a6a6"])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(category, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")
    ax.add_patch(
        Rectangle(
            (MIN_LON, MIN_LAT),
            MAX_LON - MIN_LON,
            MAX_LAT - MIN_LAT,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            linestyle="--"
        )
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(STUDY_AREA_TITLE, fontsize=14)

    legend_handles = [
        Patch(facecolor="#f4a6a6", edgecolor="black", label="Hazard area"),
        Patch(facecolor=COLOR_SAFE, edgecolor="black", label="Safe area"),
        Patch(facecolor=COLOR_OBSTACLE, edgecolor="black", label="Obstacle")
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] {output_path} を保存しました（CSV フォールバック）")


# ============================================================
# 3. 図2: mesh_overview_10m.png
# ============================================================

def plot_mesh_overview_10m(output_path, base_data):
    """
    現状の 10m メッシュのまま描く。
    全体図 + inset zoom で「格子」を見せる。
    """
    potential_v = base_data["potential_v"]
    obstacle_mask = base_data["obstacle_mask"]
    safe_mask = base_data["safe_mask"]
    hazard_passable_mask = base_data["hazard_passable_mask"]
    extent = base_data["extent"]

    # 0 obstacle, 1 safe, 2 hazard-passable
    category = np.full_like(potential_v, fill_value=np.nan, dtype=float)
    category[hazard_passable_mask] = 2
    category[safe_mask] = 1
    category[obstacle_mask] = 0

    cmap = ListedColormap([COLOR_OBSTACLE, COLOR_SAFE, COLOR_HAZARD])

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(category, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")

    ax.set_title("10 m simulation mesh overview", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # 凡例
    legend_handles = [
        Patch(facecolor=COLOR_OBSTACLE, edgecolor="black", label="Obstacle"),
        Patch(facecolor=COLOR_SAFE, edgecolor="black", label="Safe area"),
        Patch(facecolor=COLOR_HAZARD, edgecolor="black", label="Hazard / passable area")
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    # inset zoom
    H, W = potential_v.shape
    # 中央付近を適当に切り出す（必要なら調整）
    y0 = H // 2 - 20
    y1 = y0 + 40
    x0 = W // 2 - 20
    x1 = x0 + 40

    # lon / lat の対応
    lons = base_data["lons"]
    lats = base_data["lats"]
    extent_zoom = [lons[x0], lons[x1 - 1], lats[y0], lats[y1 - 1]]

    axins = ax.inset_axes([0.60, 0.05, 0.35, 0.35])
    axins.imshow(
        category[y0:y1, x0:x1],
        origin="lower",
        extent=extent_zoom,
        cmap=cmap,
        interpolation="nearest"
    )
    axins.set_title("Zoomed 10 m mesh", fontsize=9)

    # 格子線を見せる
    # minor ticks をセル境界に相当する位置へ入れる
    zoom_lons = lons[x0:x1]
    zoom_lats = lats[y0:y1]

    axins.set_xticks(zoom_lons[::5])
    axins.set_yticks(zoom_lats[::5])
    axins.tick_params(labelsize=7)

    # セル境界線
    for xx in zoom_lons:
        axins.axvline(xx, color="gray", linewidth=0.2, alpha=0.5)
    for yy in zoom_lats:
        axins.axhline(yy, color="gray", linewidth=0.2, alpha=0.5)

    # inset の範囲を元図に枠で表示
    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] {output_path} を保存しました")


# ============================================================
# 4. 図3: obstacle_mask.png
#    障害物を黒、通行可能を白、安全圏を薄い緑で重ねる
# ============================================================

def plot_obstacle_mask(output_path, base_data):
    obstacle_mask = base_data["obstacle_mask"]
    safe_mask = base_data["safe_mask"]
    extent = base_data["extent"]

    # RGB 画像を手で作る
    H, W = obstacle_mask.shape
    rgb = np.ones((H, W, 3), dtype=float)  # 初期は白（通行可能）

    # safe area を薄緑
    safe_rgb = np.array([217, 242, 217]) / 255.0  # #d9f2d9
    rgb[safe_mask] = safe_rgb

    # obstacle は黒を最優先
    rgb[obstacle_mask] = np.array([0, 0, 0]) / 255.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest")

    ax.set_title("Obstacle mask with safe area overlay", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    legend_handles = [
        Patch(facecolor=COLOR_OBSTACLE, edgecolor="black", label="Obstacle"),
        Patch(facecolor=COLOR_PASSABLE, edgecolor="black", label="Passable"),
        Patch(facecolor=COLOR_SAFE, edgecolor="black", label="Safe area")
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] {output_path} を保存しました")


# ============================================================
# 5. 図4: distance_potential_field.png
#    カラーマップ、障害物黒、安全圏緑輪郭、等高線あり
# ============================================================

def plot_distance_potential_field(output_path, base_data):
    potential_v = base_data["potential_v"]
    obstacle_mask = base_data["obstacle_mask"]
    safe_mask = base_data["safe_mask"]
    extent = base_data["extent"]
    lons = base_data["lons"]
    lats = base_data["lats"]

    V_field = compute_distance_field(potential_v, obstacle_mask)

    # 障害物は描画から除外
    V_plot = np.ma.masked_where(obstacle_mask, V_field)

    fig, ax = plt.subplots(figsize=(9, 8))

    # カラーマップ
    im = ax.imshow(
        V_plot,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest"
    )

    # 障害物を黒で重ねる
    obs_img = np.ma.masked_where(~obstacle_mask, obstacle_mask.astype(float))
    ax.imshow(
        obs_img,
        origin="lower",
        extent=extent,
        cmap=ListedColormap([COLOR_OBSTACLE]),
        interpolation="nearest",
        alpha=1.0
    )

    # 安全圏の輪郭を緑で描く
    # contour 用に 0/1 配列
    safe_int = safe_mask.astype(float)
    ax.contour(
        lons,
        lats,
        safe_int,
        levels=[0.5],
        colors="limegreen",
        linewidths=1.5
    )

    # 等高線
    finite_vals = V_field[np.isfinite(V_field)]
    vmax = np.percentile(finite_vals, 95) if finite_vals.size > 0 else 1.0
    levels = np.linspace(0, vmax, 12)

    ax.contour(
        lons,
        lats,
        np.where(np.isfinite(V_field), V_field, np.nan),
        levels=levels,
        colors="white",
        linewidths=0.5,
        alpha=0.7
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.90)
    cbar.set_label("Distance to safe area (in grid cells)")

    ax.set_title("Distance potential field to safe area", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    legend_handles = [
        Patch(facecolor=COLOR_OBSTACLE, edgecolor="black", label="Obstacle"),
        Patch(facecolor="none", edgecolor="limegreen", label="Safe-area boundary")
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] {output_path} を保存しました")


# ============================================================
# 6. 図5: macro_mdec_mevac.png
#    m_dec と m_evac、2本の間を薄く塗る
# ============================================================

def load_macro_stats(macro_csv_path):
    df = pd.read_csv(macro_csv_path)

    if "h_ext" not in df.columns:
        raise ValueError("macro_stats.csv に h_ext 列がありません。")

    # どちらの形式でも読めるようにする
    if {"m_dec", "m_evac"}.issubset(df.columns):
        m_dec_col = "m_dec"
        m_evac_col = "m_evac"
    elif {"m_dec_cum", "m_evac_cum"}.issubset(df.columns):
        m_dec_col = "m_dec_cum"
        m_evac_col = "m_evac_cum"
    else:
        raise ValueError("macro_stats.csv に m_dec/m_evac あるいは m_dec_cum/m_evac_cum がありません。")

    out = df[["h_ext", m_dec_col, m_evac_col]].copy()
    out = out.rename(columns={m_dec_col: "m_dec", m_evac_col: "m_evac"})
    out = out.sort_values("h_ext").reset_index(drop=True)
    return out


def plot_macro_mdec_mevac(output_path, macro_csv_path):
    df = load_macro_stats(macro_csv_path)

    x = df["h_ext"].to_numpy()
    y_dec = df["m_dec"].to_numpy()
    y_evac = df["m_evac"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(x, y_dec, linewidth=2.5, label=r"$m_{\mathrm{dec}}$")
    ax.plot(x, y_evac, linewidth=2.5, linestyle="--", label=r"$m_{\mathrm{evac}}$")

    # 2本の間を薄く塗る
    ax.fill_between(
        x,
        y_dec,
        y_evac,
        alpha=0.20,
        label=r"$m_{\mathrm{dec}} - m_{\mathrm{evac}}$ gap"
    )

    ax.set_xlabel(r"External field $h_{\mathrm{ext}}$")
    ax.set_ylabel("Rate")
    ax.set_title("Decision-making rate and evacuation rate", fontsize=14)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] {output_path} を保存しました")


# ============================================================
# 7. メイン
# ============================================================

def main():
    warnings.filterwarnings("ignore")
    ensure_output_dir(OUTPUT_DIR)

    base_data = load_base_map_csv(BASE_MAP_CSV)

    # 1. study_area_map.png
    plot_study_area_map(
        output_path=os.path.join(OUTPUT_DIR, "study_area_map.png"),
        db_url=DB_URL,
        use_db=USE_DB_FOR_STUDY_AREA,
        base_data=base_data
    )

    # 2. mesh_overview_10m.png
    plot_mesh_overview_10m(
        output_path=os.path.join(OUTPUT_DIR, "mesh_overview_10m.png"),
        base_data=base_data
    )

    # 3. obstacle_mask.png
    plot_obstacle_mask(
        output_path=os.path.join(OUTPUT_DIR, "obstacle_mask.png"),
        base_data=base_data
    )

    # 4. distance_potential_field.png
    plot_distance_potential_field(
        output_path=os.path.join(OUTPUT_DIR, "distance_potential_field.png"),
        base_data=base_data
    )

    # 5. macro_mdec_mevac.png
    plot_macro_mdec_mevac(
        output_path=os.path.join(OUTPUT_DIR, "macro_mdec_mevac.png"),
        macro_csv_path=MACRO_STATS_CSV
    )

    print("\n=== 完了 ===")
    print(f"出力先: {OUTPUT_DIR}")
    print("生成ファイル:")
    print(" - study_area_map.png")
    print(" - mesh_overview_10m.png")
    print(" - obstacle_mask.png")
    print(" - distance_potential_field.png")
    print(" - macro_mdec_mevac.png")


if __name__ == "__main__":
    main()