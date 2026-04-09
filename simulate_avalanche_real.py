import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sqlalchemy import create_engine
from scipy.ndimage import distance_transform_edt
from scipy.signal import convolve2d
import skfmm

# ==========================================
# 1. ベースマップの書き出し（DBからの抽出）
# ==========================================
print("▼ 1. データベースからポテンシャル場 V(x,y) を抽出しています...")
engine = create_engine('postgresql://postgres:mysecretpassword@localhost:5432/evacuation')

# PostGISから、マス目の中心座標と危険度(V)を抽出するSQL
sql = """
SELECT 
    mesh_id, 
    ST_X(ST_Centroid(geometry)) as lon, 
    ST_Y(ST_Centroid(geometry)) as lat, 
    potential_v,
    is_obstacle
FROM simulation_base_map
ORDER BY lat, lon;
"""
df = pd.read_sql(sql, con=engine)

# CSVとして書き出す（To Doリストの「CSV/メッシュデータの書き出し」完了）
df.to_csv("base_map_potential_real.csv", index=False)
print("   -> 'base_map_potential_real.csv' として保存しました。")

# ==========================================
# 2. シミュレーション空間（2Dグリッド）の構築
# ==========================================
# 経度・緯度を単純な2次元の行列インデックス(x, y)に変換
lon_unique = np.sort(df['lon'].unique())
lat_unique = np.sort(df['lat'].unique())
GRID_W = len(lon_unique)
GRID_H = len(lat_unique)

# ポテンシャル場 V の2次元配列を作成（危険=1, 安全=0）
V_field = np.zeros((GRID_H, GRID_W))
for _, row in df.iterrows():
    x_idx = np.where(lon_unique == row['lon'])[0][0]
    y_idx = np.where(lat_unique == row['lat'])[0][0]
    V_field[y_idx, x_idx] = row['potential_v']

print("▼ 1.5 障害物を定義し、FMMでポテンシャル勾配を計算しています...")

# 1. DBのデータから現実の障害物（川・建物等）のマスクを作成
obstacles = np.zeros((GRID_H, GRID_W), dtype=bool)
for _, row in df.iterrows():
    x_idx = np.where(lon_unique == row['lon'])[0][0]
    y_idx = np.where(lat_unique == row['lat'])[0][0]
    obstacles[y_idx, x_idx] = row['is_obstacle']

print(f"   -> 障害物マップを構築しました（総数: {obstacles.sum()}マス）")

# 2. FMM用の初期場を作成（安全圏=0、それ以外=1）
phi = np.ones((GRID_H, GRID_W))
phi[V_field == 0] = 0  # 既に安全圏と判定されているマスは0

# 3. 障害物を「マスク処理」してFMMに渡す
phi_masked = np.ma.MaskedArray(phi, mask=obstacles)

# 4. FMMを実行し、障害物を迂回した安全圏までの最短距離を計算
distance_field = skfmm.distance(phi_masked).data

# 5. 障害物のマスの距離（ポテンシャル V）を「無限大(∞)」に設定
distance_field[obstacles] = np.inf

# 6. ポテンシャルを正規化せず、そのままの距離（セル数）を使用する
# これにより、1マス安全な方向へ進むと必ず ΔV = -1.0 となる
V_field = distance_field

# ==========================================
# 3. マルチエージェントの設定
# ==========================================
N_AGENTS = 1000

KERNEL_SIZE = 7  # 7x7マス（約700m四方）の範囲まで影響を考慮
k_center = KERNEL_SIZE // 2
mu_kernel = np.zeros((KERNEL_SIZE, KERNEL_SIZE))

# 距離に反比例（1/r）する影響力フィルターを作成
for i in range(KERNEL_SIZE):
    for j in range(KERNEL_SIZE):
        dist = np.sqrt((i - k_center)**2 + (j - k_center)**2)
        if dist == 0:
            mu_kernel[i, j] = 1.0  # 同じマスは影響力最大
        else:
            mu_kernel[i, j] = 1.0 / dist  # 離れるほど影響力が弱まる


# 危険エリア（V>0）かつ 障害物ではない（V<無限大）座標リストを取得
danger_y, danger_x = np.where((V_field > 0) & (V_field < np.inf))

# エージェントの初期配置（危険エリアからランダムに選ぶ）
np.random.seed(42)
initial_indices = np.random.choice(len(danger_x), N_AGENTS)
agents_x = danger_x[initial_indices].astype(float)
agents_y = danger_y[initial_indices].astype(float)

# 物理パラメータ（全体共通）
# 平常時は地形の傾斜を感じない（アルファ=0）とし、純粋に外場 h_ext だけで駆動させます
alpha = 0.0      
T = 0.05         # ゆらぎを極小化し、確率的な「漏れ」を完全に防ぐ（T=0.1から変更）
h_ext = 0.0      # 初期状態
C_max = 20.0     # マスの最大収容人数

# ==========================================
# エージェントごとの個体差（ガウス分布）の生成
# ==========================================
# 同調圧力(mu)を高く、正常性バイアス(phi)も高く設定します
mu_mean, mu_std = 1.5, 0.3
phi_mean, phi_std = 1.0, 0.2

mu_array = np.random.normal(loc=mu_mean, scale=mu_std, size=N_AGENTS)
phi_bar_array = np.random.normal(loc=phi_mean, scale=phi_std, size=N_AGENTS)

# 物理的な異常挙動を防ぐためのクリッピング処理
mu_array = np.clip(mu_array, 0.0, None)
phi_bar_array = np.clip(phi_bar_array, 0.0, None)

# ※物理的な異常挙動を防ぐためのクリッピング処理
# 同調圧力がマイナス（人が多いほど逃げたくなる天邪鬼）や、
# 正常性バイアスがマイナス（理由もなく常に動き回りたがる）になるのを防ぐため、最低値を0にします。
mu_array = np.clip(mu_array, 0.0, None)
phi_bar_array = np.clip(phi_bar_array, 0.0, None)

# ==========================================
# 4. シミュレーションの更新ステップ（準静的過程）
# ==========================================
RELAXATION_STEPS = 500  # 1つの外場 h_ext に対して、系を緩和させるための内部ステップ数

def update(frame):
    global agents_x, agents_y
    
    # フレーム数に応じて外場を段階的に強くする（例: 0.00 から 0.05 刻みで上昇）
    # ※frameは0, 1, 2...と増えていくアニメーションのコマ数です
    current_h_ext = frame * 0.005 
    
    # ▼【重要】ここで内部的に時間を500ステップ進める（描画はしない）
    for step in range(RELAXATION_STEPS):
        
        # 毎ステップの人口密度を計算
        density, _, _ = np.histogram2d(agents_y, agents_x, bins=[range(GRID_H+1), range(GRID_W+1)])
        
        # 距離減衰フィルターを畳み込んで、「周囲の影響力（ローカルな平均場）」を計算
        smoothed_density = convolve2d(density, mu_kernel, mode='same', boundary='fill', fillvalue=0)
        
        for i in range(N_AGENTS):
            cx, cy = int(agents_x[i]), int(agents_y[i])
            
            # 安全圏 (V=0) にいたら動かない
            if V_field[cy, cx] == 0:
                continue
                
            # 隣接する移動候補をランダムに1つ選ぶ
            moves = [(0,1), (0,-1), (1,0), (-1,0)]
            dx, dy = moves[np.random.randint(len(moves))]
            nx, ny = cx + dx, cy + dy
            
            if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H:
                continue
                
            # 実効的リスクの計算（※ここで固定された current_h_ext を使う）
            sigma_eff_curr = (alpha + current_h_ext) * V_field[cy, cx]
            sigma_eff_next = (alpha + current_h_ext) * V_field[ny, nx]
            
            # ハミルトニアン（個体差 mu_array を使用）
            E_curr = sigma_eff_curr - mu_array[i] * smoothed_density[cy, cx]
            E_next = sigma_eff_next - mu_array[i] * smoothed_density[ny, nx]
            
            # エネルギー差分 ＋ 個体差 phi_bar_array
            delta_E = (E_next - E_curr) + phi_bar_array[i]
            
            # 熱浴法による基本確率
            P_move_base = 1.0 / (1.0 + np.exp(delta_E / T))
            
            # ソフト排他体積（渋滞ペナルティ）
            capacity_penalty = max(0.0, 1.0 - (density[ny, nx] / C_max))
            
            P_move_final = P_move_base * capacity_penalty
            
            # 確率的判定
            if np.random.rand() < P_move_final:
                density[cy, cx] -= 1
                density[ny, nx] += 1
                agents_x[i] = nx
                agents_y[i] = ny

    # ▼ 500ステップの緩和が「完了した後」のデータだけを使ってヒートマップを更新する
    current_masked_density = np.ma.masked_where(density == 0, density)
    heatmap.set_data(current_masked_density)
    
    # タイトルを更新（緩和状態であることを明記）
    ax.set_title(f"Quasi-static State (t=500) | External Field (h): {current_h_ext:.2f}")
    return heatmap,
# ==========================================
# 5. アニメーションの描画
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
# 1層目：ポテンシャル場（ハザードマップ）を背景に描画
ax.imshow(V_field, cmap='Blues', origin='lower', alpha=0.5)

# 2層目：ヒートマップ用の初期レイヤーを作成
# 初期状態の密度を計算
initial_density, _, _ = np.histogram2d(agents_y, agents_x, bins=[range(GRID_H+1), range(GRID_W+1)])
# 誰もいないマス(0人)は「透明」にするためのマスク処理
masked_density = np.ma.masked_where(initial_density == 0, initial_density)

# カラーマップ 'YlOrRd' (Yellow-Orange-Red) で描画
# vmin=1(1人で黄色), vmax=C_max(定員で濃い赤) に設定
heatmap = ax.imshow(masked_density, cmap='YlOrRd', origin='lower', alpha=0.9, vmin=1, vmax=C_max)

ax.set_xlim(0, GRID_W)
ax.set_ylim(0, GRID_H)
print("▼ 2. シミュレーション（アニメーション）を開始します...")
ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=True)
ani.save("avalanche_simulation_real.gif", writer="pillow", fps=10)
print("   -> 'avalanche_simulation_real.gif' として保存が完了しました。")
plt.show()