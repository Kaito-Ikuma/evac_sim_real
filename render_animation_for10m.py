import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os
from scipy.ndimage import maximum_filter

print("▼ 1. ベースマップと障害物を読み込んでいます...")
df_map = pd.read_csv("base_map_potential_real_10m.csv")
lon_unique = np.sort(df_map['lon'].unique())
lat_unique = np.sort(df_map['lat'].unique())
GRID_W, GRID_H = len(lon_unique), len(lat_unique)

V_field = df_map.pivot(index='lat', columns='lon', values='potential_v').values
obstacles = df_map.pivot(index='lat', columns='lon', values='is_obstacle').values
V_field = np.nan_to_num(V_field, nan=0.0)
obstacles = np.nan_to_num(obstacles, nan=0.0).astype(bool)

V_field_visual = np.where(obstacles, np.nan, V_field)

print("▼ 2. 計算結果と統計ログを取得しています...")
file_list = sorted(glob.glob("simulation_results_threshold_10m/simulation_results_threshold_10m/density_frame_*.npy"))
stats_csv = "simulation_results_threshold_10m/simulation_results_threshold_10m/macro_stats.csv"

if not file_list or not os.path.exists(stats_csv):
    print("エラー: 結果ファイルまたは macro_stats.csv が見つかりません。")
    exit()

# 統計CSVの読み込み
df_stats = pd.read_csv(stats_csv)
h_ext_list = df_stats['h_ext'].values
m_dec_list = df_stats['m_dec'].values
m_evac_list = df_stats['m_evac'].values
N_FRAMES = len(file_list)

# フレーム数とCSVの行数が一致するか確認
if N_FRAMES != len(df_stats):
    print(f"警告: .npyファイル数({N_FRAMES})とCSVの行数({len(df_stats)})が一致しません。")
    N_FRAMES = min(N_FRAMES, len(df_stats))

# ==========================================
# 【ここを追加】h_ext が 2.0 までのフレームで切り取る
# ==========================================
# # h_ext_list の中で、2.0 を超えた最初のインデックス（位置）を探す
# limit_idx = np.where(h_ext_list >= 2.0)[0]

# # もし 2.0 を超えるデータが存在すれば、そこまでで打ち切る
# if len(limit_idx) > 0:
#     N_FRAMES = limit_idx[0] + 1  # 2.0を超えた最初のフレームまで
    
#     # グラフのX軸の最大値も 2.0 に固定する（後で使う用）
#     h_ext_max_display = 2.0
#     print(f"▼ 描画範囲を h_ext=2.0 (フレーム {N_FRAMES}/{len(file_list)}) までに制限します。")
# else:
#     h_ext_max_display = max(h_ext_list) if max(h_ext_list) > 0 else 1.0

print("▼ 3. アニメーション（3画面統合）を生成しています...")
# 描画領域のセットアップ（3分割、高さの比率を 2:1:1 にする）
# --- [修正点1] レイアウトの定義 ---
layout = [['M', 'E'], ['M', 'D']]
fig, ax_dict = plt.subplot_mosaic(layout, figsize=(18, 10), gridspec_kw={'width_ratios': [2, 1]})
fig.tight_layout(pad=7.0)

ax1 = ax_dict['M'] # マップ
ax2 = ax_dict['E'] # m_evac
ax3 = ax_dict['D'] # m_dec

# ==========================================
# [上段] ヒートマップの設定
# ==========================================
ax1.imshow(V_field_visual, cmap='Greys', origin='lower', alpha=0.3)
initial_data = np.zeros((GRID_H, GRID_W))
heatmap = ax1.imshow(initial_data, cmap='Reds', origin='lower', alpha=0.7, vmin=0, vmax=1)
masked_data = np.ma.masked_where(initial_data == 0, initial_data)
title = ax1.set_title(f"Quasi-static State | External Field (h): {h_ext_list[0]:.3f}", fontsize=14)
ax1.set_xlim(0, GRID_W)
ax1.set_ylim(0, GRID_H)

# ==========================================
# [中段] m_evac (物理的な避難完了率) の設定
# ==========================================
ax2.set_title("Physical Evacuation Rate (m_evac)", fontsize=12)
ax2.set_ylabel("Evacuation Rate", fontsize=10)
ax2.set_xlim(0, max(h_ext_list) if max(h_ext_list) > 0 else 1.0)
# ax2.set_xlim(0, h_ext_max_display)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True)
ax2.plot(h_ext_list, m_evac_list, color='gray', linestyle='--', alpha=0.5) # 未来の軌跡
line_evac, = ax2.plot([], [], color='red', linewidth=2)
point_evac, = ax2.plot([], [], marker='o', color='red', markersize=8)

# ==========================================
# [下段] m_dec (意思決定完了率) の設定
# ==========================================
ax3.set_title("Decision Making Rate (m_dec)", fontsize=12)
ax3.set_xlabel("External Field (h_ext)", fontsize=12)
ax3.set_ylabel("Decision Rate", fontsize=10)
ax3.set_xlim(0, max(h_ext_list) if max(h_ext_list) > 0 else 1.0)
# ax3.set_xlim(0, h_ext_max_display)
ax3.set_ylim(-0.05, 1.05)
ax3.grid(True)
ax3.plot(h_ext_list, m_dec_list, color='gray', linestyle='--', alpha=0.5) # 未来の軌跡
line_dec, = ax3.plot([], [], color='blue', linewidth=2)
point_dec, = ax3.plot([], [], marker='o', color='blue', markersize=8)

# ==========================================
# アニメーション更新関数
# ==========================================
def update(frame_idx):
    # 1. ヒートマップの更新
    density_data = np.load(file_list[frame_idx])
    fat_density = maximum_filter(density_data, size=3)
    fat_density = np.where(fat_density > 0, 1.0, 0.0)
    masked_density = np.ma.masked_where(fat_density == 0, density_data)
    heatmap.set_data(masked_density)
    
    h_ext = h_ext_list[frame_idx]
    title.set_text(f"Quasi-static State | External Field (h): {h_ext:.3f}")
    
    # 2. m_evac グラフの更新
    line_evac.set_data(h_ext_list[:frame_idx+1], m_evac_list[:frame_idx+1])
    point_evac.set_data([h_ext_list[frame_idx]], [m_evac_list[frame_idx]])

    # 3. m_dec グラフの更新
    line_dec.set_data(h_ext_list[:frame_idx+1], m_dec_list[:frame_idx+1])
    point_dec.set_data([h_ext_list[frame_idx]], [m_dec_list[frame_idx]])
    
    return heatmap, title, line_evac, point_evac, line_dec, point_dec

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=200, blit=True)

output_filename = "simulation_results_threshold_10m/mpi_avalanche_with_dec_and_evac.gif"
ani.save(output_filename, writer='pillow', fps=5, dpi=200)
print(f"▼ 完了！ '{output_filename}' を保存しました。")