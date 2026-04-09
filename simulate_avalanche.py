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
    potential_v
FROM simulation_base_map
ORDER BY lat, lon;
"""
df = pd.read_sql(sql, con=engine)

# CSVとして書き出す（To Doリストの「CSV/メッシュデータの書き出し」完了）
df.to_csv("base_map_potential.csv", index=False)
print("   -> 'base_map_potential.csv' として保存しました。")

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

# 1. 障害物（川や建物）のマスクを作成する
# ここではテストとして、マップ中央に「川」を作り、真ん中に「橋」を架けます
obstacles = np.zeros((GRID_H, GRID_W), dtype=bool)
river_x = GRID_W // 2
obstacles[:, river_x] = True  # 縦一列をすべて障害物（川）にする
obstacles[GRID_H//2 - 2 : GRID_H//2 + 2, river_x] = False  # 中央の数マスだけ通行可能（橋）にする

# 2. FMM用の初期場を作成（安全圏=0、それ以外=1）
phi = np.ones((GRID_H, GRID_W))
phi[V_field == 0] = 0  # 既に安全圏と判定されているマスは0

# 3. 障害物を「マスク処理」してFMMに渡す
phi_masked = np.ma.MaskedArray(phi, mask=obstacles)

# 4. FMMを実行し、障害物を迂回した安全圏までの最短距離を計算
distance_field = skfmm.distance(phi_masked).data

# 5. 障害物のマスの距離（ポテンシャル V）を「無限大(∞)」に設定
distance_field[obstacles] = np.inf

# 6. ポテンシャルを0〜1に正規化（無限大のマスを除外して最大値で割る）
max_dist = np.max(distance_field[~obstacles])
V_field = distance_field / max_dist
V_field[obstacles] = np.inf  # 正規化後に再度、無限大を設定

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

# 物理パラメータ
alpha = 2.0      # 地形ポテンシャルへのベース感受性
mu = 0.5         # 同調圧力（人が多い方に引かれる強さ）
T = 1.0          # 温度（行動のランダムさ・ゆらぎ）
h_ext = 0.0      # 外場（初期状態ではゼロ）
phi_bar = 0.5    # 正常性バイアス（移動を開始するための心理的障壁）

# ==========================================
# 4. シミュレーションの更新ステップ（Metropolis法）
# ==========================================
def update(frame):
    global agents_x, agents_y, h_ext
    
    # 時間経過とともに外場（避難指示や危機感）が強くなる設定
    if frame > 20:
        h_ext += 0.05 
        
    # 各マスの現在の人口密度を計算（同調圧力の計算用）
    density, _, _ = np.histogram2d(agents_y, agents_x, bins=[range(GRID_H+1), range(GRID_W+1)])
    
    # 距離減衰フィルターを畳み込んで、「周囲の影響力（ローカルな平均場）」を計算する
    smoothed_density = convolve2d(density, mu_kernel, mode='same', boundary='fill', fillvalue=0)
    
    for i in range(N_AGENTS):
        # 1. まず現在位置を取得
        cx, cy = int(agents_x[i]), int(agents_y[i])
        
        # 2. もし既に安全圏 (V=0) にいたら、それ以上動かない（吸収状態）
        if V_field[cy, cx] == 0:
            continue
            
        # 3. 隣接する移動候補をランダムに1つ選ぶ（上下左右）
        moves = [(0,1), (0,-1), (1,0), (-1,0)]
        dx, dy = moves[np.random.randint(len(moves))]
        nx, ny = cx + dx, cy + dy
        
        # 4. 実効的リスク sigma_eff の計算
        # 外場 h_ext が高いほど、危険地帯 V_field のリスクを強く感じるように定式化
        sigma_eff_curr = (alpha + h_ext) * V_field[cy, cx]
        sigma_eff_next = (alpha + h_ext) * V_field[ny, nx]
        
        # ハミルトニアン（各座標でのエネルギー）
        E_curr = sigma_eff_curr - mu * smoothed_density[cy, cx]
        E_next = sigma_eff_next - mu * smoothed_density[ny, nx]
        
        # エネルギー差分 ＋ 正常性バイアス（越えるべき障壁）
        delta_E = (E_next - E_curr) + phi_bar
        
        # 6. 熱浴法（Heat Bath / Glauber Dynamics）による遷移確率の計算
        P_move = 1.0 / (1.0 + np.exp(delta_E / T))
        
        # 確率的判定
        if np.random.rand() < P_move:
            # 移動を承認
            agents_x[i] = nx
            agents_y[i] = ny
            
        # 5. 現在位置と移動先が決まった後で、エネルギー差分を計算（※ここで smoothed_density を使う）
        current_E = alpha * V_field[cy, cx] - mu * smoothed_density[cy, cx]
        next_E = alpha * V_field[ny, nx] - mu * smoothed_density[ny, nx] - h_ext
        
        delta_E = next_E - current_E
        
        # 6. Metropolis判定
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            # 移動を承認
            agents_x[i] = nx
            agents_y[i] = ny

    # 描画の更新
    scat.set_offsets(np.c_[agents_x, agents_y])
    ax.set_title(f"Time Step: {frame} | External Field (h): {h_ext:.2f}")
    return scat,

# ==========================================
# 5. アニメーションの描画
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
# ポテンシャル場（ハザードマップ）を背景に描画
ax.imshow(V_field, cmap='Blues', origin='lower', alpha=0.5)
# エージェントを赤い点で描画
scat = ax.scatter(agents_x, agents_y, c='red', s=5, alpha=0.8)

ax.set_xlim(0, GRID_W)
ax.set_ylim(0, GRID_H)

print("▼ 2. シミュレーション（アニメーション）を開始します...")
ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=True)
ani.save("avalanche_simulation.gif", writer="pillow", fps=10)
print("   -> 'avalanche_simulation.gif' として保存が完了しました。")
plt.show()