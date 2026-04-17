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

# ==========================================
# 3.5 ローカル配列化 + ghost + 可変緩和 用の補助設定
# ==========================================
HALO_MOVE = 1                       # 移動は上下1マスなので密度の halo は 1
HALO_GHOST = int(np.ceil(R_radius)) # 意思決定近傍 R_radius 分の ghost 幅
MAX_RELAX_STEPS = 5000
MIN_RELAX_STEPS = 200
STABLE_STEPS_REQUIRED = 20
EVENT_TOL = 5

local_h = my_y_max - my_y_min + 1


def y_to_local(y_global: int) -> int:
    """
    グローバル y 座標 -> この rank の density_local 上の行番号
    owned 領域は [HALO_MOVE, HALO_MOVE + local_h - 1]
    """
    return int(y_global - my_y_min) + HALO_MOVE


def pack_full_payload(agent_id, x, y, mu, phi, s,
                      dens_before, dens_after,
                      step_decided, evac_time, owner_rank):
    """
    rank 間 migration 用 payload
    空なら shape=(0, 10) の配列を返す
    """
    if len(agent_id) == 0:
        return np.empty((0, 10), dtype=np.float64)

    return np.column_stack([
        agent_id.astype(np.float64),
        x.astype(np.float64),
        y.astype(np.float64),
        mu.astype(np.float64),
        phi.astype(np.float64),
        s.astype(np.float64),
        dens_before.astype(np.float64),
        dens_after.astype(np.float64),
        step_decided.astype(np.float64),
        evac_time.astype(np.float64),
    ])


def append_full_payload(payload,
                        agent_id, x, y, mu, phi, s,
                        dens_before, dens_after,
                        step_decided, evac_time, owner_rank,
                        density_local):
    """
    受信 payload をローカル配列へ追加し、owned 行の density_local も更新する
    """
    if payload is None:
        return (agent_id, x, y, mu, phi, s,
                dens_before, dens_after, step_decided, evac_time, owner_rank,
                density_local)

    payload = np.asarray(payload)
    if payload.size == 0:
        return (agent_id, x, y, mu, phi, s,
                dens_before, dens_after, step_decided, evac_time, owner_rank,
                density_local)

    if payload.ndim == 1:
        payload = payload.reshape(1, -1)

    recv_id          = payload[:, 0].astype(np.int32)
    recv_x           = payload[:, 1].astype(np.int32)
    recv_y           = payload[:, 2].astype(np.int32)
    recv_mu          = payload[:, 3].astype(np.float64)
    recv_phi         = payload[:, 4].astype(np.float64)
    recv_s           = payload[:, 5].astype(np.int8)
    recv_dens_before = payload[:, 6].astype(np.float64)
    recv_dens_after  = payload[:, 7].astype(np.float64)
    recv_step_dec    = payload[:, 8].astype(np.int64)
    recv_evac        = payload[:, 9].astype(np.int64)
    recv_owner       = np.full(len(recv_id), rank, dtype=np.int32)

    agent_id    = np.concatenate([agent_id, recv_id])
    x           = np.concatenate([x, recv_x])
    y           = np.concatenate([y, recv_y])
    mu          = np.concatenate([mu, recv_mu])
    phi         = np.concatenate([phi, recv_phi])
    s           = np.concatenate([s, recv_s])
    dens_before = np.concatenate([dens_before, recv_dens_before])
    dens_after  = np.concatenate([dens_after, recv_dens_after])
    step_decided = np.concatenate([step_decided, recv_step_dec])
    evac_time   = np.concatenate([evac_time, recv_evac])
    owner_rank  = np.concatenate([owner_rank, recv_owner])

    # 受信したエージェントは、この rank の owned 領域に入っているはず
    for xi, yi in zip(recv_x, recv_y):
        density_local[y_to_local(int(yi)), int(xi)] += 1

    return (agent_id, x, y, mu, phi, s,
            dens_before, dens_after, step_decided, evac_time, owner_rank,
            density_local)


def build_ghost_payload(x, y, s):
    """
    意思決定近傍参照用の ghost payload を上下隣接 rank 向けに作る
    列は [x, y, s]
    """
    empty = np.empty((0, 3), dtype=np.int32)

    if len(x) == 0:
        return empty, empty

    up_mask = (y <= my_y_min + HALO_GHOST - 1)
    down_mask = (y >= my_y_max - HALO_GHOST + 1)

    up_payload = empty
    down_payload = empty

    if np.any(up_mask):
        up_payload = np.column_stack([
            x[up_mask].astype(np.int32),
            y[up_mask].astype(np.int32),
            s[up_mask].astype(np.int32),
        ])

    if np.any(down_mask):
        down_payload = np.column_stack([
            x[down_mask].astype(np.int32),
            y[down_mask].astype(np.int32),
            s[down_mask].astype(np.int32),
        ])

    return up_payload, down_payload


def exchange_density_halo(density_local):
    """
    density_local の owned 境界 1 行だけを上下隣 rank と交換
    """
    top_owned = HALO_MOVE
    bottom_owned = HALO_MOVE + local_h - 1

    # 上側 ghost row
    if rank > 0:
        recv_top = comm.sendrecv(
            sendobj=density_local[top_owned, :].copy(),
            dest=rank - 1,
            source=rank - 1
        )
        density_local[0, :] = recv_top
    else:
        density_local[0, :] = 0

    # 下側 ghost row
    if rank < size - 1:
        recv_bottom = comm.sendrecv(
            sendobj=density_local[bottom_owned, :].copy(),
            dest=rank + 1,
            source=rank + 1
        )
        density_local[bottom_owned + 1, :] = recv_bottom
    else:
        density_local[bottom_owned + 1, :] = 0

    return density_local


# ==========================================
# 3.6 dict -> NumPy配列へ置換
# ==========================================
local_init_mask = (agents_y >= my_y_min) & (agents_y <= my_y_max)

agent_id = np.arange(N_AGENTS, dtype=np.int32)[local_init_mask]
x = agents_x[local_init_mask].astype(np.int32)
y = agents_y[local_init_mask].astype(np.int32)
mu = mu_array[local_init_mask].astype(np.float64)
phi = phi_bar_array[local_init_mask].astype(np.float64)

s = -np.ones(len(agent_id), dtype=np.int8)
local_density_before_move = np.zeros(len(agent_id), dtype=np.float64)
local_density_after_move = np.zeros(len(agent_id), dtype=np.float64)
step_when_decided = -np.ones(len(agent_id), dtype=np.int64)
evacuation_time = -np.ones(len(agent_id), dtype=np.int64)
owner_rank = np.full(len(agent_id), rank, dtype=np.int32)

# owned rows + 上下 halo 1 行
density_local = np.zeros((local_h + 2 * HALO_MOVE, GRID_W), dtype=np.int32)

for xi, yi in zip(x, y):
    density_local[y_to_local(int(yi)), int(xi)] += 1


# ==========================================
# 4. メインシミュレーションループ（統合版）
# ==========================================
NUM_FRAMES = 400

output_dir = "simulation_results_threshold_10m"
history_log = []
if rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    print("=== MPI並列計算を開始します（統合版） ===")

global_step_counter = 0

for frame in range(NUM_FRAMES):
    current_h_ext = frame * 0.005

    stable_count = 0
    steps_used_this_frame = MAX_RELAX_STEPS

    for step in range(MAX_RELAX_STEPS):
        abs_step = global_step_counter + step

        # ==========================================
        # [A] ghost agent 交換（意思決定用）
        # ==========================================
        send_up_ghost, send_down_ghost = build_ghost_payload(x, y, s)

        ghost_from_up = np.empty((0, 3), dtype=np.int32)
        ghost_from_down = np.empty((0, 3), dtype=np.int32)

        if rank > 0:
            ghost_from_up = comm.sendrecv(
                sendobj=send_up_ghost,
                dest=rank - 1,
                source=rank - 1
            )
            ghost_from_up = np.asarray(ghost_from_up, dtype=np.int32)

        if rank < size - 1:
            ghost_from_down = comm.sendrecv(
                sendobj=send_down_ghost,
                dest=rank + 1,
                source=rank + 1
            )
            ghost_from_down = np.asarray(ghost_from_down, dtype=np.int32)

        ghost_parts = []
        if ghost_from_up.size > 0:
            ghost_parts.append(ghost_from_up)
        if ghost_from_down.size > 0:
            ghost_parts.append(ghost_from_down)

        if len(ghost_parts) > 0:
            ghost_all = np.vstack(ghost_parts)
            ghost_x = ghost_all[:, 0].astype(np.int32)
            ghost_y = ghost_all[:, 1].astype(np.int32)
            ghost_s = ghost_all[:, 2].astype(np.int8)
            cand_x = np.concatenate([x, ghost_x])
            cand_y = np.concatenate([y, ghost_y])
            cand_s = np.concatenate([s, ghost_s])
        else:
            cand_x = x.copy()
            cand_y = y.copy()
            cand_s = s.copy()

        # ==========================================
        # [B] 意思決定更新（owned agent のみ）
        # ==========================================
        local_flip_count = 0

        for k in range(len(x)):
            dx = cand_x - x[k]
            dy = cand_y - y[k]
            distances = np.sqrt(dx.astype(np.float64)**2 + dy.astype(np.float64)**2)

            valid_mask = (distances > 0.0) & (distances <= R_radius)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                H_i = current_h_ext - phi[k]
            else:
                valid_distances = distances[valid_indices]
                K_r = 1.0 / valid_distances
                p_ij = K_r / np.sum(K_r)

                actual_c = min(c_neighbors, len(valid_indices))
                sampled_indices = np.random.choice(
                    valid_indices,
                    size=actual_c,
                    replace=False,
                    p=p_ij
                )

                std_dev = sigma_J / np.sqrt(c_neighbors)
                J_ij = np.random.normal(
                    loc=mu[k] / c_neighbors,
                    scale=std_dev,
                    size=actual_c
                )

                sum_interaction = np.sum(J_ij * cand_s[sampled_indices])
                H_i = current_h_ext - phi[k] + sum_interaction

            P_plus1 = 1.0 / (1.0 + np.exp(-2.0 * H_i / T_dec))

            old_s = s[k]
            s[k] = 1 if (np.random.rand() < P_plus1) else -1

            if old_s != s[k]:
                local_flip_count += 1

            if old_s == -1 and s[k] == 1 and step_when_decided[k] == -1:
                step_when_decided[k] = abs_step

        # ==========================================
        # [C] density halo 交換（移動用）
        # ==========================================
        density_local = exchange_density_halo(density_local)

        # ==========================================
        # [D] 移動更新（density は増分更新）
        # ==========================================
        local_move_count = 0

        for k in range(len(x)):
            cx = int(x[k])
            cy = int(y[k])
            cy_loc = y_to_local(cy)

            local_density_before_move[k] = density_local[cy_loc, cx]

            if s[k] == -1:
                local_density_after_move[k] = density_local[cy_loc, cx]
                continue

            if V_field[cy, cx] == 0:
                if evacuation_time[k] == -1:
                    evacuation_time[k] = abs_step
                local_density_after_move[k] = density_local[cy_loc, cx]
                continue

            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            mdx, mdy = moves[np.random.randint(4)]
            nx = cx + mdx
            ny = cy + mdy

            if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H:
                local_density_after_move[k] = density_local[cy_loc, cx]
                continue

            ny_loc = y_to_local(ny)
            if ny_loc < 0 or ny_loc >= density_local.shape[0]:
                local_density_after_move[k] = density_local[cy_loc, cx]
                continue

            delta_V = V_field[ny, nx] - V_field[cy, cx]
            P_move_base = 1.0 / (1.0 + np.exp(delta_V / T))
            capacity_penalty = max(0.0, 1.0 - (density_local[ny_loc, nx] / C_max))
            P_move_final = P_move_base * capacity_penalty

            if np.random.rand() < P_move_final:
                density_local[cy_loc, cx] -= 1
                density_local[ny_loc, nx] += 1
                x[k] = nx
                y[k] = ny
                local_move_count += 1

            ax = int(x[k])
            ay = int(y[k])
            ay_loc = y_to_local(ay)

            if 0 <= ay_loc < density_local.shape[0]:
                local_density_after_move[k] = density_local[ay_loc, ax]
            else:
                local_density_after_move[k] = 0.0

            if V_field[ay, ax] == 0 and evacuation_time[k] == -1:
                evacuation_time[k] = abs_step

        # ==========================================
        # [E] 境界越え migration
        # ==========================================
        send_up_mask = (y < my_y_min)
        send_down_mask = (y > my_y_max)
        keep_mask = (~send_up_mask) & (~send_down_mask)

        payload_up = pack_full_payload(
            agent_id[send_up_mask], x[send_up_mask], y[send_up_mask],
            mu[send_up_mask], phi[send_up_mask], s[send_up_mask],
            local_density_before_move[send_up_mask],
            local_density_after_move[send_up_mask],
            step_when_decided[send_up_mask], evacuation_time[send_up_mask],
            owner_rank[send_up_mask]
        )

        payload_down = pack_full_payload(
            agent_id[send_down_mask], x[send_down_mask], y[send_down_mask],
            mu[send_down_mask], phi[send_down_mask], s[send_down_mask],
            local_density_before_move[send_down_mask],
            local_density_after_move[send_down_mask],
            step_when_decided[send_down_mask], evacuation_time[send_down_mask],
            owner_rank[send_down_mask]
        )

        # 自分に残る agent だけ保持
        agent_id = agent_id[keep_mask]
        x = x[keep_mask]
        y = y[keep_mask]
        mu = mu[keep_mask]
        phi = phi[keep_mask]
        s = s[keep_mask]
        local_density_before_move = local_density_before_move[keep_mask]
        local_density_after_move = local_density_after_move[keep_mask]
        step_when_decided = step_when_decided[keep_mask]
        evacuation_time = evacuation_time[keep_mask]
        owner_rank = owner_rank[keep_mask]

        recv_from_up = np.empty((0, 10), dtype=np.float64)
        recv_from_down = np.empty((0, 10), dtype=np.float64)

        if rank > 0:
            recv_from_up = comm.sendrecv(
                sendobj=payload_up,
                dest=rank - 1,
                source=rank - 1
            )

        if rank < size - 1:
            recv_from_down = comm.sendrecv(
                sendobj=payload_down,
                dest=rank + 1,
                source=rank + 1
            )

        (agent_id, x, y, mu, phi, s,
         local_density_before_move, local_density_after_move,
         step_when_decided, evacuation_time, owner_rank,
         density_local) = append_full_payload(
            recv_from_up,
            agent_id, x, y, mu, phi, s,
            local_density_before_move, local_density_after_move,
            step_when_decided, evacuation_time, owner_rank,
            density_local
        )

        (agent_id, x, y, mu, phi, s,
         local_density_before_move, local_density_after_move,
         step_when_decided, evacuation_time, owner_rank,
         density_local) = append_full_payload(
            recv_from_down,
            agent_id, x, y, mu, phi, s,
            local_density_before_move, local_density_after_move,
            step_when_decided, evacuation_time, owner_rank,
            density_local
        )

        # ==========================================
        # [F] 可変緩和の停止判定
        # ==========================================
        global_events = comm.allreduce(local_flip_count + local_move_count, op=MPI.SUM)

        if step >= MIN_RELAX_STEPS and global_events < EVENT_TOL:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= STABLE_STEPS_REQUIRED:
            steps_used_this_frame = step + 1
            break

    global_step_counter += steps_used_this_frame

    # ==========================================
    # 5. フレーム保存フェーズ
    # ==========================================
    my_agents_info = []
    for k in range(len(agent_id)):
        my_agents_info.append({
            'agent_id': int(agent_id[k]),
            'x': int(x[k]),
            'y': int(y[k]),
            's': int(s[k]),
            'phi': float(phi[k]),
            'mu': float(mu[k]),
            'local_density_before_move': float(local_density_before_move[k]),
            'local_density_after_move': float(local_density_after_move[k]),
            'step_when_decided': int(step_when_decided[k]),
            'evacuation_time': int(evacuation_time[k]),
            'rank': int(owner_rank[k]),
            'is_safe': 1 if V_field[int(y[k]), int(x[k])] == 0 else 0
        })

    gathered_agents = comm.gather(my_agents_info, root=0)

    # 保存時だけ全体密度を復元
    density_full_local = np.zeros((GRID_H, GRID_W), dtype=np.int32)
    density_full_local[my_y_min:my_y_max + 1, :] = density_local[HALO_MOVE:HALO_MOVE + local_h, :]

    global_density = np.zeros_like(density_full_local)
    comm.Allreduce(density_full_local, global_density, op=MPI.SUM)

    my_decided = int(np.sum(s == 1))
    total_decided = comm.allreduce(my_decided, op=MPI.SUM)
    m_dec = total_decided / N_AGENTS

    my_safe = int(np.sum(V_field[y.astype(np.int32), x.astype(np.int32)] == 0))
    total_safe = comm.allreduce(my_safe, op=MPI.SUM)
    m_evac = total_safe / N_AGENTS

    if rank == 0:
        all_agents_flat = [agent for sublist in gathered_agents for agent in sublist]
        df_agents = pd.DataFrame(all_agents_flat)

        agent_csv_path = os.path.join(output_dir, f"agents_frame_{frame:03d}.csv")
        df_agents.to_csv(agent_csv_path, index=False)

        density_path = os.path.join(output_dir, f"density_frame_{frame:03d}.npy")
        np.save(density_path, global_density)

        history_log.append([current_h_ext, m_dec, m_evac, steps_used_this_frame])
        print(
            f"[Frame {frame:03d}/{NUM_FRAMES-1}] "
            f"h_ext={current_h_ext:.3f}, "
            f"m_dec={m_dec:.4f}, m_evac={m_evac:.4f}, "
            f"steps={steps_used_this_frame} -> 保存: {density_path}"
        )

print(f"Time: {time.time() - start}s")

if rank == 0:
    df_log = pd.DataFrame(
        history_log,
        columns=['h_ext', 'm_dec', 'm_evac', 'steps_used_this_frame']
    )
    df_log.to_csv(os.path.join(output_dir, "macro_stats.csv"), index=False)
    print("=== 全計算プロセスが完了しました ===")