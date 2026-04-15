import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

print("▼ 1. ベースマップと障害物を読み込んでいます...")
df = pd.read_csv("base_map_potential_real.csv")
lon_unique = np.sort(df['lon'].unique())
lat_unique = np.sort(df['lat'].unique())
GRID_W, GRID_H = len(lon_unique), len(lat_unique)

V_field = np.zeros((GRID_H, GRID_W))
obstacles = np.zeros((GRID_H, GRID_W), dtype=bool)
for _, row in df.iterrows():
    x_idx = np.where(lon_unique == row['lon'])[0][0]
    y_idx = np.where(lat_unique == row['lat'])[0][0]
    V_field[y_idx, x_idx] = row['potential_v']
    obstacles[y_idx, x_idx] = row['is_obstacle']

V_field_visual = np.where(obstacles, np.nan, V_field)

print("▼ 2. 計算結果ファイルを取得しています...")
file_list = sorted(glob.glob("simulation_results_threshold/density_frame_*.npy"))
if not file_list:
    print("エラー: 結果ファイルが見つかりません。")
    exit()

N_FRAMES = len(file_list)
N_AGENTS = 1000

print("▼ 3. 各フレームの避難完了率 m を事前計算しています...")
m_list = []
h_ext_list = []

for i, f in enumerate(file_list):
    density_data = np.load(f)
    # V_field が 0 のマス（安全圏）にいる人数の合計を計算
    safe_count = np.sum(density_data[V_field == 0])
    m = safe_count / N_AGENTS
    
    # ※シミュレーション側の刻み幅に合わせてください（例: 0.015）
    h_ext = i * 0.015  
    
    m_list.append(m)
    h_ext_list.append(h_ext)

print("▼ 4. アニメーション（2画面統合）を生成しています...")
# 描画領域のセットアップ（上下に2分割、高さの比率を 2:1 にする）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), gridspec_kw={'height_ratios': [2, 1]})
fig.tight_layout(pad=4.0)

# ==========================================
# [上のグラフ] ヒートマップの設定
# ==========================================
ax1.imshow(V_field_visual, cmap='Blues', origin='lower', alpha=0.5)
initial_data = np.load(file_list[0])
masked_data = np.ma.masked_where(initial_data == 0, initial_data)
heatmap = ax1.imshow(masked_data, cmap='YlOrRd', origin='lower', alpha=0.9, vmin=1, vmax=20.0)
title = ax1.set_title(f"Quasi-static State | External Field (h): {h_ext_list[0]:.3f}", fontsize=14)
ax1.set_xlim(0, GRID_W)
ax1.set_ylim(0, GRID_H)

# ==========================================
# [下のグラフ] m(t) の相転移グラフの設定
# ==========================================
ax2.set_title("Phase Transition: External Field vs Evacuation Rate", fontsize=14)
ax2.set_xlabel("External Field (h_ext)", fontsize=12)
ax2.set_ylabel("Evacuation Rate (m)", fontsize=12)
ax2.set_xlim(0, max(h_ext_list) if max(h_ext_list) > 0 else 1.0)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True)

# 背景に全行程の「薄い点線」を描画（未来の軌道を先に示しておく）
ax2.plot(h_ext_list, m_list, color='gray', linestyle='--', alpha=0.5)

# 現在のフレームまでの軌跡を描画する太い赤線
line, = ax2.plot([], [], color='red', linewidth=2)
# 現在のフレームの位置を示す赤い点（マーカー）
point, = ax2.plot([], [], marker='o', color='red', markersize=8)

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
    
    # 2. グラフの更新
    # 0番目から現在のフレームまでのデータを切り出して線を描く
    line.set_data(h_ext_list[:frame_idx+1], m_list[:frame_idx+1])
    # 現在の点だけをプロットする
    point.set_data([h_ext_list[frame_idx]], [m_list[frame_idx]])
    
    return heatmap, title, line, point

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=200, blit=True)

output_filename = "mpi_avalanche_with_graph_threshold.gif"
ani.save(output_filename, writer='pillow', fps=5)
print(f"▼ 完了！ '{output_filename}' を保存しました。")