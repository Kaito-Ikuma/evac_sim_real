import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from sqlalchemy import create_engine
from scipy.ndimage import distance_transform_edt
from scipy.signal import convolve2d
import skfmm
from mpi4py import MPI
import os 
import time; start = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



# ==========================================
# 1. ベースマップの書き出し（DBからの抽出）
# ==========================================
# print("▼ 1. データベースからポテンシャル場 V(x,y) を抽出しています...")
# engine = create_engine('postgresql://postgres:mysecretpassword@localhost:5432/evacuation')

# # PostGISから、マス目の中心座標と危険度(V)を抽出するSQL
# sql = """
# SELECT 
#     mesh_id, 
#     ST_X(ST_Centroid(geometry)) as lon, 
#     ST_Y(ST_Centroid(geometry)) as lat, 
#     potential_v,
#     is_obstacle
# FROM simulation_base_map
# ORDER BY lat, lon;
# """
# df = pd.read_sql(sql, con=engine)

# # CSVとして書き出す（To Doリストの「CSV/メッシュデータの書き出し」完了）
# df.to_csv("base_map_potential_real.csv", index=False)
# print("   -> 'base_map_potential_real.csv' として保存しました。")

print("▼ 1. CSVからポテンシャル場 V(x,y) を抽出しています...")
df = pd.read_csv("base_map_potential_real_10m.csv")

# ==========================================
# 2. シミュレーション空間（2Dグリッド）の構築
# ==========================================
# 経度・緯度を単純な2次元の行列インデックス(x, y)に変換
lon_unique = np.sort(df['lon'].unique())
lat_unique = np.sort(df['lat'].unique())
GRID_W = len(lon_unique)
GRID_H = len(lat_unique)

# 自分の担当する Y座標 の範囲を計算
chunk_size = GRID_H // size
my_y_min = rank * chunk_size
# 最後のランクは余りも含めて最後まで担当する
my_y_max = (rank + 1) * chunk_size - 1 if rank != size - 1 else GRID_H - 1

# V_field などは巨大ではないので、全員が全体マップを持っておき、
# 自分の担当エリア（my_y_min <= y <= my_y_max）にいるエージェントだけを計算対象とする

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
N_AGENTS = 2000

KERNEL_SIZE = 51  # 51x51マス（約5000m四方）の範囲まで影響を考慮
# k_center = KERNEL_SIZE // 2
# mu_kernel = np.zeros((KERNEL_SIZE, KERNEL_SIZE))

# # 距離に反比例（1/r）する影響力フィルターを作成
# for i in range(KERNEL_SIZE):
#     for j in range(KERNEL_SIZE):
#         dist = np.sqrt((i - k_center)**2 + (j - k_center)**2)
#         if dist == 0:
#             mu_kernel[i, j] = 1.0  # 同じマスは影響力最大
#         else:
#             mu_kernel[i, j] = 1.0 / dist  # 離れるほど影響力が弱まる

# mu_kernel /= np.sum(mu_kernel)


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
T = 0.01         # ゆらぎを極小化し、確率的な「漏れ」を完全に防ぐ（T=0.1から変更）
h_ext = 0.0      # 初期状態
C_max = 50.0     # マスの最大収容人数
R_radius = 2.0      # 候補集合の半径 R (距離)
c_neighbors = 50     # サンプルする近傍数 c
sigma_J = 0.5        # 相互作用 J_ij のばらつき (標準偏差ではなく分散 σ_J^2 の平方根)
T_dec = 0.01          # 決定温度 (有効ノイズ)

# ==========================================
# エージェントごとの個体差（ガウス分布）の生成
# ==========================================
# 同調圧力(mu)を高く、正常性バイアス(phi)も高く設定します
mu_mean, mu_std = 2.0, 0.1
phi_mean, phi_std = 1.0, 0.05

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

my_agents = []

# N_AGENTS人全員をチェックし、自分の担当エリア(my_y_min 〜 my_y_max)にいる人だけを確保
for i in range(N_AGENTS):
    if my_y_min <= agents_y[i] <= my_y_max:
        agent_dict = {
            'x': agents_x[i],
            'y': agents_y[i],
            'mu': mu_array[i],
            'phi': phi_bar_array[i],
            's': -1
        }
        my_agents.append(agent_dict)

# ==========================================
# 4. メインシミュレーションループ（MPI・保存のみ）
# ==========================================
RELAXATION_STEPS = 5000
NUM_FRAMES = 400  # 外場 h_ext を変化させる回数

# 結果保存用のディレクトリを作成（Rank 0 のみ実行）
output_dir = "simulation_results_threshold"
history_log = []
if rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    print("=== MPI並列計算を開始します ===")

# 外場 h_ext を段階的に上げながら計算を進める
for frame in range(NUM_FRAMES):
    current_h_ext = frame * 0.005
    
    # ▼【内部で時間を進める（準静的過程）】
    for step in range(RELAXATION_STEPS):

        # ==========================================
        # ==========================================
        # [フェーズ0] 全エージェント情報の共有 (Allgather)
        # ==========================================
        # 自分が担当しているエージェントの「座標とスピン」だけを抽出
        my_simple_agents = [{'x': a['x'], 'y': a['y'], 's': a['s']} for a in my_agents]

        # MPIで全コアのデータをかき集める（リストのリストが返ってくる）
        gathered_agents = comm.allgather(my_simple_agents)

        # リストを平坦化（Flatten）して、全エージェントのリストを作成
        all_visible_agents = [agent for sublist in gathered_agents for agent in sublist]

        # numpy配列化（フェーズ1のベクトル計算用）
        all_x = np.array([a['x'] for a in all_visible_agents])
        all_y = np.array([a['y'] for a in all_visible_agents])
        all_s = np.array([a['s'] for a in all_visible_agents])

        # ==========================================
        # [フェーズ1 & 2] 動的ネットワーク構築と意思決定 (Ising Update)
        # ==========================================
        # ※ my_agents (自分の担当) のみ更新する。ゴーストは参照用。
        for i, agent in enumerate(my_agents):
            
            # --- 1. 候補集合 Ω_i(t) の作成 ---
            dx = all_x - agent['x']
            dy = all_y - agent['y']
            distances = np.sqrt(dx**2 + dy**2)

            # 自身(距離0)を除外 & 半径 R 以内を抽出
            valid_mask = (distances > 0) & (distances <= R_radius)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                # 候補が誰もいない場合は孤独（外場と正常性バイアスのみ）
                H_i = current_h_ext - agent['phi']
            else:
                # --- 2. 確率 p_ij ∝ K(r_ij) の計算 ---
                valid_distances = distances[valid_indices]
                # カーネル K(r) = 1/r の例（減衰関数）
                K_r = 1.0 / valid_distances 
                p_ij = K_r / np.sum(K_r)  # 確率の合計を1に正規化

                # --- 3. c個の近傍をサンプルし N_i(t) を作る ---
                # 候補が c より少ない場合は、いる人数だけサンプルする
                actual_c = min(c_neighbors, len(valid_indices))
                sampled_indices = np.random.choice(
                    valid_indices, size=actual_c, replace=False, p=p_ij
                )

                # --- 4. 相互作用 J_ij の生成と H_i の計算 ---
                # J_ij ~ N(μ_i / c, σ_J^2 / c)
                # 分散が σ_J^2/c なので、標準偏差(scale)は √(σ_J^2 / c) = σ_J / √c となる
                std_dev = sigma_J / np.sqrt(c_neighbors)
                J_ij = np.random.normal(
                    loc=agent['mu'] / c_neighbors, 
                    scale=std_dev, 
                    size=actual_c
                )

                # Σ J_ij * s_j を計算
                sum_interaction = np.sum(J_ij * all_s[sampled_indices])
                H_i = current_h_ext - agent['phi'] + sum_interaction

            # --- 5. 確率的スピン更新 P(s = +1) ---
            P_plus1 = 1.0 / (1.0 + np.exp(-2.0 * H_i / T_dec))
            
            if np.random.rand() < P_plus1:
                agent['s'] = 1
            else:
                agent['s'] = -1
        
        # --- [フェーズ0] 密度場とスピン場の計算 ---
        my_agents_y = [a['y'] for a in my_agents]
        my_agents_x = [a['x'] for a in my_agents]
        
        # 物理移動用の人口密度
        density, _, _ = np.histogram2d(my_agents_y, my_agents_x, bins=[range(GRID_H+1), range(GRID_W+1)])
        density = np.ascontiguousarray(density)

        # # 意思決定用の局所スピン場（自分の担当エリアの s_i をマッピング）
        # local_spin_grid = np.zeros((GRID_H, GRID_W))
        # for a in my_agents:
        #     local_spin_grid[int(a['y']), int(a['x'])] += a['s']

        # # MPI: スピン場を全コアで合算・共有し、完全なグローバルスピン場を作成
        # global_spin_grid = np.zeros_like(local_spin_grid)
        # comm.Allreduce(local_spin_grid, global_spin_grid, op=MPI.SUM)

        # # --- [フェーズ1] 相互作用場の計算とのり代交換 ---
        # # 空間カーネル K を用いた畳み込み：これが sum J_ij * K * s_j に相当します
        # interaction_field = convolve2d(global_spin_grid, mu_kernel, mode='same')

        # 物理移動のためののり代（Halo）の交換 (densityのみ)
        if rank > 0:
            comm.Sendrecv(density[my_y_min, :], dest=rank-1, recvbuf=density[my_y_min-1, :], source=rank-1)
        if rank < size - 1:
            comm.Sendrecv(density[my_y_max, :], dest=rank+1, recvbuf=density[my_y_max+1, :], source=rank+1)

        # --- [フェーズ2] レイヤー1：意思決定の更新 (Ising Dynamics) ---
        # for agent in my_agents:
        #     cx, cy = int(agent['x']), int(agent['y'])
            
        #     # 局所場 H_i の計算：外場 - 正常性バイアス + 同調圧力(相互作用場)
        #     H_i = current_h_ext - agent['phi'] + agent['mu'] * interaction_field[cy, cx]
            
        #     # スピンの更新 (ステップ関数)
        #     if H_i > 0:
        #         agent['s'] = 1
        #     elif H_i < 0:
        #         agent['s'] = -1

        # # --- [フェーズ3] レイヤー2：物理空間の移動 (Floor Field Dynamics) ---
        for agent in my_agents:
            if agent['s'] == -1:
                continue # 意思決定が -1 の人は動かない

            cx, cy = int(agent['x']), int(agent['y'])
            if V_field[cy, cx] == 0:
                continue # すでに安全圏
                
            moves = [(0,1), (0,-1), (1,0), (-1,0)]
            dx, dy = moves[np.random.randint(len(moves))]
            nx, ny = cx + dx, cy + dy
            
            if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H:
                continue
                
            # 移動確率は「物理的なポテンシャル差 (delta_V)」のみで決まる
            delta_V = V_field[ny, nx] - V_field[cy, cx]
            
            # 温度 T による揺らぎと渋滞ペナルティ
            P_move_base = 1.0 / (1.0 + np.exp(delta_V / T))
            capacity_penalty = max(0.0, 1.0 - (density[ny, nx] / C_max))
            P_move_final = P_move_base * capacity_penalty
            
            if np.random.rand() < P_move_final:
                density[cy, cx] -= 1
                density[ny, nx] += 1
                agent['x'] = nx
                agent['y'] = ny

        # --- [フェーズ4] マクロ統計量 m_dec と m_evac の算出 ---
        # 意思決定完了率 m_dec
        my_spin_sum = sum(a['s'] for a in my_agents)
        # total_spin_sum = comm.allreduce(my_spin_sum, op=MPI.SUM)
        m_dec = my_spin_sum / N_AGENTS

        # 避難完了率 m_evac
        my_safe_count = sum(1 for a in my_agents if V_field[int(a['y']), int(a['x'])] == 0)
        # total_safe_count = comm.allreduce(my_safe_count, op=MPI.SUM)
        m_evac = my_safe_count / N_AGENTS
                
        # --- [フェーズ4] エージェントのマイグレーション（境界越え） ---
        agents_to_send_up = []
        agents_to_send_down = []
        agents_to_keep = []

        for agent in my_agents:
            if agent['y'] < my_y_min:
                agents_to_send_up.append(agent)
            elif agent['y'] > my_y_max:
                agents_to_send_down.append(agent)
            else:
                agents_to_keep.append(agent)

        my_agents = agents_to_keep

        if rank > 0:
            incoming_from_up = comm.sendrecv(agents_to_send_up, dest=rank-1, source=rank-1)
            my_agents.extend(incoming_from_up)

        if rank < size - 1:
            incoming_from_down = comm.sendrecv(agents_to_send_down, dest=rank+1, source=rank+1)
            my_agents.extend(incoming_from_down)

    # ==========================================
    # 5. 緩和完了後の結果保存フェーズ (MPI Reduce)
    # ==========================================

    # 自分のコアが担当しているエージェントの情報をリスト化
    my_agents_info = []
    for a in my_agents:
        my_agents_info.append({
            'agent_id': a['id'], # もし初期化時にIDを振っていれば
            'x': a['x'],
            'y': a['y'],
            's': a['s'],         # 意思 (1:避難, -1:待機)
            'phi': a['phi'],     # 正常性バイアス
            'is_safe': 1 if V_field[int(a['y']), int(a['x'])] == 0 else 0
        })
    # 全コアのエージェント情報をRank 0にかき集める
    gathered_agents = comm.gather(my_agents_info, root=0)

    density_contig = np.ascontiguousarray(density)
    global_density = np.zeros_like(density_contig)
    
    # 全コアのdensityを足し合わせてマップ全体を復元
    comm.Reduce(density_contig, global_density, op=MPI.SUM, root=0)
    
    # 意思決定完了率 m_dec (スピン s = 1 の人の割合)
    my_decided = sum(1 for a in my_agents if a['s'] == 1)
    total_decided = comm.allreduce(my_decided, op=MPI.SUM)
    m_dec = total_decided / N_AGENTS

    # 避難完了率 m_evac (V_field == 0 にいる人の割合)
    my_safe = sum(1 for a in my_agents if V_field[int(a['y']), int(a['x'])] == 0)
    total_safe = comm.allreduce(my_safe, op=MPI.SUM)
    m_evac = total_safe / N_AGENTS
    
    if rank == 0:
        # リストのリストを平坦化（Flatten）
        all_agents_flat = [agent for sublist in gathered_agents for agent in sublist]
        df_agents = pd.DataFrame(all_agents_flat)
        
        # 毎フレーム、エージェント全員の状態をCSVとして保存
        agent_csv_path = os.path.join(output_dir, f"agents_frame_{frame:03d}.csv")
        df_agents.to_csv(agent_csv_path, index=False)
        filename = os.path.join(output_dir, f"density_frame_{frame:03d}.npy")
        np.save(filename, global_density)
        history_log.append([current_h_ext, m_dec, m_evac]) 
        print(f"[Frame {frame:02d}/{NUM_FRAMES-1}] 緩和完了 (h_ext={current_h_ext:.2f}, m_dec={m_dec:.2f}, m_evac={m_evac:.2f}) -> 保存: {filename}")
print(f"Time: {time.time() - start}s")
if rank == 0:
    df_log = pd.DataFrame(history_log, columns=['h_ext', 'm_dec', 'm_evac'])
    df_log.to_csv(os.path.join(output_dir, "macro_stats.csv"), index=False)
    print("=== 全計算プロセスが完了しました ===")