import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os

print("▼ 1. ベースマップと障害物を読み込んでいます...")
df_map = pd.read_csv("base_map_potential_real_10m.csv")
lon_unique = np.sort(df_map['lon'].unique())
lat_unique = np.sort(df_map['lat'].unique())
GRID_W, GRID_H = len(lon_unique), len(lat_unique)

V_field = np.zeros((GRID_H, GRID_W))
obstacles = np.zeros((GRID_H, GRID_W), dtype=bool)
for _, row in df_map.iterrows():
    x_idx = np.where(lon_unique == row['lon'])[0][0]
    y_idx = np.where(lat_unique == row['lat'])[0][0]
    V_field[y_idx, x_idx] = row['potential_v']
    obstacles[y_idx, x_idx] = row['is_obstacle']

V_field_visual = np.where(obstacles, np.nan, V_field)

print("▼ 2. 計算結果と統計ログを取得しています...")
file_list = sorted(glob.glob("simulation_results_threshold_10m/simulation_results_threshold/density_frame_*.npy"))
stats_csv = "simulation_results_threshold_10m/simulation_results_threshold/macro_stats.csv"

if not file_list or not os.path.exists(stats_csv):
    print("エラー: 結果ファイルまたは macro_stats.csv が見つかりません。")
    exit()

# 統計CSVの読み込み
df_stats = pd.read_csv(stats_csv)
h_ext_list = df_stats['h_ext'].values
m_dec_list = df_stats['m_dec'].values
m_evac_list = df_stats['m_evac'].values
N_FRAMES = len(file_list)

print("▼ 3. アニメーション（3画面統合）を生成しています...")
# 描画領域のセットアップ（3分割、高さの比率を 2:1:1 にする）
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 15), gridspec_kw={'height_ratios': [2, 1, 1]})
fig.tight_layout(pad=5.0)

# ==========================================
# [上段] ヒートマップの設定
# ==========================================
ax1.imshow(V_field_visual, cmap='Blues', origin='lower', alpha=0.5)
initial_data = np.load(file_list[0])
masked_data = np.ma.masked_where(initial_data == 0, initial_data)
heatmap = ax1.imshow(masked_data, cmap='Reds', origin='lower', alpha=1.0, vmin=0.0, vmax=3.0)
title = ax1.set_title(f"Quasi-static State | External Field (h): {h_ext_list[0]:.3f}", fontsize=14)
ax1.set_xlim(0, GRID_W)
ax1.set_ylim(0, GRID_H)

# ==========================================
# [中段] m_evac (物理的な避難完了率) の設定
# ==========================================
ax2.set_title("Physical Evacuation Rate (m_evac)", fontsize=12)
ax2.set_ylabel("Evacuation Rate", fontsize=10)
ax2.set_xlim(0, max(h_ext_list) if max(h_ext_list) > 0 else 1.0)
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
    masked_density = np.ma.masked_where(density_data == 0, density_data)
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
ani.save(output_filename, writer='pillow', fps=5, dpi=150)
print(f"▼ 完了！ '{output_filename}' を保存しました。")