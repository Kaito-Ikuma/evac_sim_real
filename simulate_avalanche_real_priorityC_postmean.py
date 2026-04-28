
import numpy as np
import pandas as pd
import skfmm
from mpi4py import MPI
import os
import time
from collections import deque

PRIORITY = "C_POSTMEAN"
ENABLE_B = True
ENABLE_C = True

start = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("▼ 1. CSVからポテンシャル場 V(x,y) を抽出しています...")
df = pd.read_csv("base_map_potential_real_10m.csv")
lon_unique = np.sort(df['lon'].unique())
lat_unique = np.sort(df['lat'].unique())
GRID_W = len(lon_unique)
GRID_H = len(lat_unique)

chunk_size = GRID_H // size
my_y_min = rank * chunk_size
my_y_max = (rank + 1) * chunk_size - 1 if rank != size - 1 else GRID_H - 1
local_h = my_y_max - my_y_min + 1

V_field = np.zeros((GRID_H, GRID_W))
obstacles = np.zeros((GRID_H, GRID_W), dtype=bool)
lon_to_ix = {v: i for i, v in enumerate(lon_unique)}
lat_to_iy = {v: i for i, v in enumerate(lat_unique)}
for _, row in df.iterrows():
    x_idx = lon_to_ix[row['lon']]
    y_idx = lat_to_iy[row['lat']]
    V_field[y_idx, x_idx] = row['potential_v']
    obstacles[y_idx, x_idx] = bool(row['is_obstacle'])

print("▼ 1.5 障害物を定義し、FMMでポテンシャル勾配を計算しています...")
phi0 = np.ones((GRID_H, GRID_W))
phi0[V_field == 0] = 0
phi_masked = np.ma.MaskedArray(phi0, mask=obstacles)
distance_field = skfmm.distance(phi_masked).data
distance_field[obstacles] = np.inf
V_field = distance_field

N_AGENTS = 2000
T = 0.01
C_max = 50.0
R_radius = 2.0
c_neighbors = 50
sigma_J = 0.5
T_dec = 0.01
H_FINE_MIN = 0.80
H_FINE_MAX = 1.20
H_FINE_STEP = 0.005
H_COARSE_LEFT_MIN = 0.00
H_COARSE_LEFT_MAX = 0.70
H_COARSE_LEFT_STEP = 0.05
H_TRANS_LEFT_MIN = 0.70
H_TRANS_LEFT_MAX = 0.80
H_TRANS_LEFT_STEP = 0.01
H_TRANS_RIGHT_MIN = 1.20
H_TRANS_RIGHT_MAX = 1.30
H_TRANS_RIGHT_STEP = 0.01
H_COARSE_RIGHT_MIN = 1.30
H_COARSE_RIGHT_MAX = 1.50
H_COARSE_RIGHT_STEP = 0.05
mu_mean, mu_std = 2.0, 0.1
phi_mean, phi_std = 1.0, 0.05
HALO_MOVE = 1
HALO_GHOST = int(np.ceil(R_radius))
MAX_RELAX_STEPS = 5000
MIN_RELAX_STEPS = 200
STABLE_STEPS_REQUIRED = 20
EVENT_TOL = 5
SOC_TRACE_MAX_AGENTS = 128
SOC_TRACE_PRE_WINDOW = 64
UPSTREAM_MAX_V = 5.0


def arange_inclusive(start, stop, step):
    n = int(np.floor((stop - start) / step + 1e-12))
    values = start + step * np.arange(n + 1, dtype=np.float64)
    values = values[values <= stop + 1e-12]
    return values


def build_h_schedule():
    parts = [
        arange_inclusive(H_COARSE_LEFT_MIN, H_COARSE_LEFT_MAX, H_COARSE_LEFT_STEP),
        arange_inclusive(H_TRANS_LEFT_MIN + H_TRANS_LEFT_STEP, H_TRANS_LEFT_MAX, H_TRANS_LEFT_STEP),
        arange_inclusive(H_FINE_MIN + H_FINE_STEP, H_FINE_MAX, H_FINE_STEP),
        arange_inclusive(H_TRANS_RIGHT_MIN + H_TRANS_RIGHT_STEP, H_TRANS_RIGHT_MAX, H_TRANS_RIGHT_STEP),
        arange_inclusive(H_COARSE_RIGHT_MIN + H_COARSE_RIGHT_STEP, H_COARSE_RIGHT_MAX, H_COARSE_RIGHT_STEP),
    ]
    h_values = np.concatenate(parts)
    # 重複や浮動小数誤差を避ける
    h_values = np.unique(np.round(h_values, 12))
    return h_values


H_VALUES = build_h_schedule()
NUM_FRAMES = len(H_VALUES)


def y_to_local(y_global: int) -> int:
    return int(y_global - my_y_min) + HALO_MOVE


def count_downhill_neighbors(cx: int, cy: int) -> int:
    curV = V_field[cy, cx]
    count = 0
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nx, ny = cx + dx, cy + dy
        if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H:
            continue
        if obstacles[ny, nx]:
            continue
        if np.isfinite(V_field[ny, nx]) and V_field[ny, nx] < curV:
            count += 1
    return count


def build_upstream_mask(max_v: float = 5.0):
    return (V_field > 0) & (V_field <= max_v) & np.isfinite(V_field)


UPSTREAM_MASK = build_upstream_mask(UPSTREAM_MAX_V)

danger_y, danger_x = np.where((V_field > 0) & (V_field < np.inf))
np.random.seed(42)
initial_indices = np.random.choice(len(danger_x), N_AGENTS)
agents_x = danger_x[initial_indices].astype(float)
agents_y = danger_y[initial_indices].astype(float)
mu_array = np.clip(np.random.normal(loc=mu_mean, scale=mu_std, size=N_AGENTS), 0.0, None)
phi_bar_array = np.clip(np.random.normal(loc=phi_mean, scale=phi_std, size=N_AGENTS), 0.0, None)

BASE_FIELDS = [
    'agent_id', 'x', 'y', 'mu', 'phi', 's',
    'dens_before', 'dens_after', 'step_decided', 'evac_time'
]
A_FIELDS = [
    'decision_frame', 'decision_h', 'evacuation_frame', 'evacuation_h',
    'I_soc_at_decision', 'H_at_decision', 'local_m_at_decision', 'n_neighbors_at_decision'
]
B_FIELDS = [
    'V_at_decision', 'density_at_decision', 'g_down_at_decision',
    'post_g_down_sum', 'post_g_down_count',
    'post_density_sum', 'post_density_count',
    'post_no_progress_frames', 'post_stagnated_frames'
]
C_FIELDS = [
    'pre_I_soc_sum', 'pre_I_soc_sq_sum', 'pre_I_soc_count', 'pre_I_soc_max', 'pre_I_soc_min'
]
PAYLOAD_FIELDS = BASE_FIELDS + A_FIELDS + B_FIELDS + C_FIELDS
FIELD_INDEX = {name: i for i, name in enumerate(PAYLOAD_FIELDS)}
INT_FIELDS = {
    'agent_id', 'x', 'y', 's', 'step_decided', 'evac_time',
    'decision_frame', 'evacuation_frame', 'n_neighbors_at_decision',
    'g_down_at_decision', 'post_g_down_count', 'post_density_count',
    'post_no_progress_frames', 'post_stagnated_frames', 'pre_I_soc_count'
}


def pack_payload(mask, arrays):
    n = int(np.sum(mask))
    if n == 0:
        return np.empty((0, len(PAYLOAD_FIELDS)), dtype=np.float64)
    cols = [np.asarray(arrays[name][mask], dtype=np.float64) for name in PAYLOAD_FIELDS]
    return np.column_stack(cols)


def append_payload(payload, arrays, density_local):
    if payload is None:
        return arrays, density_local
    payload = np.asarray(payload)
    if payload.size == 0:
        return arrays, density_local
    if payload.ndim == 1:
        payload = payload.reshape(1, -1)
    recv = {}
    for name in PAYLOAD_FIELDS:
        col = payload[:, FIELD_INDEX[name]]
        recv[name] = col.astype(np.int64) if name in INT_FIELDS else col.astype(np.float64)
    for name in arrays:
        arrays[name] = np.concatenate([arrays[name], recv[name]])
    for xi, yi in zip(recv['x'].astype(np.int32), recv['y'].astype(np.int32)):
        density_local[y_to_local(int(yi)), int(xi)] += 1
    return arrays, density_local


def build_ghost_payload(x, y, s):
    empty = np.empty((0, 3), dtype=np.int32)
    if len(x) == 0:
        return empty, empty
    up_mask = (y <= my_y_min + HALO_GHOST - 1)
    down_mask = (y >= my_y_max - HALO_GHOST + 1)
    up_payload = empty if not np.any(up_mask) else np.column_stack([x[up_mask].astype(np.int32), y[up_mask].astype(np.int32), s[up_mask].astype(np.int32)])
    down_payload = empty if not np.any(down_mask) else np.column_stack([x[down_mask].astype(np.int32), y[down_mask].astype(np.int32), s[down_mask].astype(np.int32)])
    return up_payload, down_payload


def exchange_density_halo(density_local):
    top_owned = HALO_MOVE
    bottom_owned = HALO_MOVE + local_h - 1
    if rank > 0:
        recv_top = comm.sendrecv(sendobj=density_local[top_owned, :].copy(), dest=rank - 1, source=rank - 1)
        density_local[0, :] = recv_top
    else:
        density_local[0, :] = 0
    if rank < size - 1:
        recv_bottom = comm.sendrecv(sendobj=density_local[bottom_owned, :].copy(), dest=rank + 1, source=rank + 1)
        density_local[bottom_owned + 1, :] = recv_bottom
    else:
        density_local[bottom_owned + 1, :] = 0
    return density_local


def local_arrays_subset(arrays, mask):
    return {k: v[mask] for k, v in arrays.items()}


local_init_mask = (agents_y >= my_y_min) & (agents_y <= my_y_max)
local_n = int(np.sum(local_init_mask))
arrays = {
    'agent_id': np.arange(N_AGENTS, dtype=np.int64)[local_init_mask],
    'x': agents_x[local_init_mask].astype(np.int64),
    'y': agents_y[local_init_mask].astype(np.int64),
    'mu': mu_array[local_init_mask].astype(np.float64),
    'phi': phi_bar_array[local_init_mask].astype(np.float64),
    's': -np.ones(local_n, dtype=np.int64),
    'dens_before': np.zeros(local_n, dtype=np.float64),
    'dens_after': np.zeros(local_n, dtype=np.float64),
    'step_decided': -np.ones(local_n, dtype=np.int64),
    'evac_time': -np.ones(local_n, dtype=np.int64),
    'decision_frame': -np.ones(local_n, dtype=np.int64),
    'decision_h': np.full(local_n, np.nan),
    'evacuation_frame': -np.ones(local_n, dtype=np.int64),
    'evacuation_h': np.full(local_n, np.nan),
    'I_soc_at_decision': np.full(local_n, np.nan),
    'H_at_decision': np.full(local_n, np.nan),
    'local_m_at_decision': np.full(local_n, np.nan),
    'n_neighbors_at_decision': -np.ones(local_n, dtype=np.int64),
    'V_at_decision': np.full(local_n, np.nan),
    'density_at_decision': np.full(local_n, np.nan),
    'g_down_at_decision': -np.ones(local_n, dtype=np.int64),
    'post_g_down_sum': np.zeros(local_n, dtype=np.float64),
    'post_g_down_count': np.zeros(local_n, dtype=np.int64),
    'post_density_sum': np.zeros(local_n, dtype=np.float64),
    'post_density_count': np.zeros(local_n, dtype=np.int64),
    'post_no_progress_frames': np.zeros(local_n, dtype=np.int64),
    'post_stagnated_frames': np.zeros(local_n, dtype=np.int64),
    'pre_I_soc_sum': np.zeros(local_n, dtype=np.float64),
    'pre_I_soc_sq_sum': np.zeros(local_n, dtype=np.float64),
    'pre_I_soc_count': np.zeros(local_n, dtype=np.int64),
    'pre_I_soc_max': np.full(local_n, -np.inf),
    'pre_I_soc_min': np.full(local_n, np.inf),
}

density_local = np.zeros((local_h + 2 * HALO_MOVE, GRID_W), dtype=np.int32)
for xi, yi in zip(arrays['x'], arrays['y']):
    density_local[y_to_local(int(yi)), int(xi)] += 1

sampled_agent_ids = set(np.linspace(0, N_AGENTS - 1, SOC_TRACE_MAX_AGENTS, dtype=int).tolist())
ring_buffers = {int(aid): deque(maxlen=SOC_TRACE_PRE_WINDOW) for aid in arrays['agent_id'] if int(aid) in sampled_agent_ids}
social_trace_rows_local = []

output_dir = "simulation_results_threshold_10m_c_postmean"
history_log = []
if rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== MPI並列計算を開始します（Priority {PRIORITY}） ===")
    print(f"h-schedule: total {NUM_FRAMES} points, min={H_VALUES.min():.3f}, max={H_VALUES.max():.3f}")

global_step_counter = 0
prev_total_decided_cum = 0
prev_total_evacuated_cum = 0

for frame, current_h_ext in enumerate(H_VALUES):
    frame_start_step = global_step_counter
    stable_count = 0
    steps_used_this_frame = MAX_RELAX_STEPS
    local_exit_events = []

    for step in range(MAX_RELAX_STEPS):
        abs_step = global_step_counter + step

        send_up_ghost, send_down_ghost = build_ghost_payload(arrays['x'], arrays['y'], arrays['s'])
        ghost_from_up = np.empty((0, 3), dtype=np.int32)
        ghost_from_down = np.empty((0, 3), dtype=np.int32)
        if rank > 0:
            ghost_from_up = np.asarray(comm.sendrecv(sendobj=send_up_ghost, dest=rank - 1, source=rank - 1), dtype=np.int32)
        if rank < size - 1:
            ghost_from_down = np.asarray(comm.sendrecv(sendobj=send_down_ghost, dest=rank + 1, source=rank + 1), dtype=np.int32)
        ghost_parts = []
        if ghost_from_up.size > 0:
            ghost_parts.append(ghost_from_up)
        if ghost_from_down.size > 0:
            ghost_parts.append(ghost_from_down)
        if ghost_parts:
            ghost_all = np.vstack(ghost_parts)
            cand_x = np.concatenate([arrays['x'], ghost_all[:, 0].astype(np.int64)])
            cand_y = np.concatenate([arrays['y'], ghost_all[:, 1].astype(np.int64)])
            cand_s = np.concatenate([arrays['s'], ghost_all[:, 2].astype(np.int64)])
        else:
            cand_x = arrays['x'].copy()
            cand_y = arrays['y'].copy()
            cand_s = arrays['s'].copy()

        local_flip_count = 0
        for k in range(len(arrays['x'])):
            dx = cand_x - arrays['x'][k]
            dy = cand_y - arrays['y'][k]
            distances = np.sqrt(dx.astype(np.float64)**2 + dy.astype(np.float64)**2)
            valid_mask = (distances > 0.0) & (distances <= R_radius)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                I_soc_current = 0.0
                actual_c = 0
                local_m_current = np.nan
                H_i = current_h_ext - arrays['phi'][k]
            else:
                valid_distances = distances[valid_indices]
                K_r = 1.0 / valid_distances
                p_ij = K_r / np.sum(K_r)
                actual_c = min(c_neighbors, len(valid_indices))
                sampled_indices = np.random.choice(valid_indices, size=actual_c, replace=False, p=p_ij)
                std_dev = sigma_J / np.sqrt(c_neighbors)
                J_ij = np.random.normal(loc=arrays['mu'][k] / c_neighbors, scale=std_dev, size=actual_c)
                I_soc_current = float(np.sum(J_ij * cand_s[sampled_indices]))
                local_m_current = float(np.mean(cand_s[sampled_indices]))
                H_i = current_h_ext - arrays['phi'][k] + I_soc_current

            aid = int(arrays['agent_id'][k])
            if aid in ring_buffers:
                ring_buffers[aid].append((abs_step, frame, current_h_ext, float(I_soc_current), float(H_i), int(arrays['s'][k]), int(arrays['x'][k]), int(arrays['y'][k])))
            if arrays['step_decided'][k] == -1:
                arrays['pre_I_soc_sum'][k] += I_soc_current
                arrays['pre_I_soc_sq_sum'][k] += I_soc_current ** 2
                arrays['pre_I_soc_count'][k] += 1
                arrays['pre_I_soc_max'][k] = max(arrays['pre_I_soc_max'][k], I_soc_current)
                arrays['pre_I_soc_min'][k] = min(arrays['pre_I_soc_min'][k], I_soc_current)

            P_plus1 = 1.0 / (1.0 + np.exp(-2.0 * H_i / T_dec))
            old_s = arrays['s'][k]
            arrays['s'][k] = 1 if (np.random.rand() < P_plus1) else -1
            if old_s != arrays['s'][k]:
                local_flip_count += 1

            if old_s == -1 and arrays['s'][k] == 1 and arrays['step_decided'][k] == -1:
                arrays['step_decided'][k] = abs_step
                arrays['decision_frame'][k] = frame
                arrays['decision_h'][k] = current_h_ext
                arrays['I_soc_at_decision'][k] = I_soc_current
                arrays['H_at_decision'][k] = H_i
                arrays['local_m_at_decision'][k] = local_m_current
                arrays['n_neighbors_at_decision'][k] = actual_c
                cx = int(arrays['x'][k]); cy = int(arrays['y'][k])
                arrays['V_at_decision'][k] = V_field[cy, cx]
                arrays['density_at_decision'][k] = density_local[y_to_local(cy), cx]
                arrays['g_down_at_decision'][k] = count_downhill_neighbors(cx, cy)
                if aid in ring_buffers:
                    for rec in ring_buffers[aid]:
                        social_trace_rows_local.append((aid, *rec, 'pre_window'))
                    social_trace_rows_local.append((aid, abs_step, frame, current_h_ext, float(I_soc_current), float(H_i), int(arrays['s'][k]), int(arrays['x'][k]), int(arrays['y'][k]), 'decision'))

        density_local = exchange_density_halo(density_local)
        local_move_count = 0
        for k in range(len(arrays['x'])):
            cx = int(arrays['x'][k]); cy = int(arrays['y'][k])
            cy_loc = y_to_local(cy)
            arrays['dens_before'][k] = density_local[cy_loc, cx]
            if arrays['s'][k] == -1:
                arrays['dens_after'][k] = density_local[cy_loc, cx]
                continue

            if arrays['step_decided'][k] >= 0 and arrays['evac_time'][k] == -1:
                g_down = count_downhill_neighbors(cx, cy)
                arrays['post_g_down_sum'][k] += g_down
                arrays['post_g_down_count'][k] += 1
                arrays['post_density_sum'][k] += density_local[cy_loc, cx]
                arrays['post_density_count'][k] += 1

            if V_field[cy, cx] == 0:
                if arrays['evac_time'][k] == -1:
                    arrays['evac_time'][k] = abs_step
                    arrays['evacuation_frame'][k] = frame
                    arrays['evacuation_h'][k] = current_h_ext
                    local_exit_events.append((abs_step, frame, current_h_ext, int(arrays['agent_id'][k]), cx, cy))
                arrays['dens_after'][k] = density_local[cy_loc, cx]
                continue

            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            mdx, mdy = moves[np.random.randint(4)]
            nx = cx + mdx; ny = cy + mdy
            prev_x, prev_y = cx, cy
            prev_V = V_field[cy, cx]
            moved = False
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                ny_loc = y_to_local(ny)
                if 0 <= ny_loc < density_local.shape[0]:
                    delta_V = V_field[ny, nx] - V_field[cy, cx]
                    P_move_base = 1.0 / (1.0 + np.exp(delta_V / T))
                    capacity_penalty = max(0.0, 1.0 - (density_local[ny_loc, nx] / C_max))
                    P_move_final = P_move_base * capacity_penalty
                    if np.random.rand() < P_move_final:
                        density_local[cy_loc, cx] -= 1
                        density_local[ny_loc, nx] += 1
                        arrays['x'][k] = nx
                        arrays['y'][k] = ny
                        local_move_count += 1
                        moved = True

            ax = int(arrays['x'][k]); ay = int(arrays['y'][k]); ay_loc = y_to_local(ay)
            arrays['dens_after'][k] = density_local[ay_loc, ax] if 0 <= ay_loc < density_local.shape[0] else 0.0

            if arrays['step_decided'][k] >= 0 and arrays['evac_time'][k] == -1:
                cur_V = V_field[ay, ax]
                disp_frame = np.sqrt((ax - prev_x) ** 2 + (ay - prev_y) ** 2)
                dV = prev_V - cur_V
                if disp_frame < 1e-12:
                    arrays['post_stagnated_frames'][k] += 1
                if dV <= 0.0:
                    arrays['post_no_progress_frames'][k] += 1

            if V_field[ay, ax] == 0 and arrays['evac_time'][k] == -1:
                arrays['evac_time'][k] = abs_step
                arrays['evacuation_frame'][k] = frame
                arrays['evacuation_h'][k] = current_h_ext
                local_exit_events.append((abs_step, frame, current_h_ext, int(arrays['agent_id'][k]), ax, ay))

        send_up_mask = (arrays['y'] < my_y_min)
        send_down_mask = (arrays['y'] > my_y_max)
        keep_mask = (~send_up_mask) & (~send_down_mask)
        payload_up = pack_payload(send_up_mask, arrays)
        payload_down = pack_payload(send_down_mask, arrays)
        arrays = local_arrays_subset(arrays, keep_mask)
        recv_from_up = np.empty((0, len(PAYLOAD_FIELDS)), dtype=np.float64)
        recv_from_down = np.empty((0, len(PAYLOAD_FIELDS)), dtype=np.float64)
        if rank > 0:
            recv_from_up = comm.sendrecv(sendobj=payload_up, dest=rank - 1, source=rank - 1)
        if rank < size - 1:
            recv_from_down = comm.sendrecv(sendobj=payload_down, dest=rank + 1, source=rank + 1)
        arrays, density_local = append_payload(recv_from_up, arrays, density_local)
        arrays, density_local = append_payload(recv_from_down, arrays, density_local)

        global_events = comm.allreduce(local_flip_count + local_move_count, op=MPI.SUM)
        if step >= MIN_RELAX_STEPS and global_events < EVENT_TOL:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count >= STABLE_STEPS_REQUIRED:
            steps_used_this_frame = step + 1
            break

    global_step_counter += steps_used_this_frame
    frame_end_step = global_step_counter

    gathered_agents = comm.gather({k: arrays[k].copy() for k in arrays}, root=0)
    gathered_exit_events = comm.gather(local_exit_events, root=0)
    gathered_social_trace = comm.gather(social_trace_rows_local, root=0)
    social_trace_rows_local = []

    density_full_local = np.zeros((GRID_H, GRID_W), dtype=np.int32)
    density_full_local[my_y_min:my_y_max + 1, :] = density_local[HALO_MOVE:HALO_MOVE + local_h, :]
    global_density = np.zeros_like(density_full_local)
    comm.Allreduce(density_full_local, global_density, op=MPI.SUM)

    total_decided_inst = comm.allreduce(int(np.sum(arrays['s'] == 1)), op=MPI.SUM)
    m_dec_inst = total_decided_inst / N_AGENTS
    total_decided_cum = comm.allreduce(int(np.sum(arrays['step_decided'] >= 0)), op=MPI.SUM)
    m_dec_cum = total_decided_cum / N_AGENTS
    total_evac_inst = comm.allreduce(int(np.sum(V_field[arrays['y'].astype(np.int32), arrays['x'].astype(np.int32)] == 0)), op=MPI.SUM)
    m_evac_inst = total_evac_inst / N_AGENTS
    total_evac_cum = comm.allreduce(int(np.sum(arrays['evac_time'] >= 0)), op=MPI.SUM)
    m_evac_cum = total_evac_cum / N_AGENTS

    A_dec_frame = total_decided_cum - prev_total_decided_cum
    J_out_frame = total_evac_cum - prev_total_evacuated_cum
    Q_transport = total_decided_cum - total_evac_cum
    q_transport = Q_transport / N_AGENTS
    prev_total_decided_cum = total_decided_cum
    prev_total_evacuated_cum = total_evac_cum

    rho_up = float(global_density[UPSTREAM_MASK].mean()) if np.any(UPSTREAM_MASK) else np.nan
    local_Q_up = 0
    for k in range(len(arrays['x'])):
        if arrays['step_decided'][k] >= 0 and arrays['evac_time'][k] < 0 and UPSTREAM_MASK[int(arrays['y'][k]), int(arrays['x'][k])]:
            local_Q_up += 1
    Q_up = comm.allreduce(local_Q_up, op=MPI.SUM)

    if rank == 0:
        all_agent_rows = []
        for part in gathered_agents:
            n = len(part['agent_id'])
            for idx in range(n):
                c_pre = int(part['pre_I_soc_count'][idx])
                pre_mean_soc = float(part['pre_I_soc_sum'][idx] / c_pre) if c_pre > 0 else np.nan
                pre_var_soc = float(part['pre_I_soc_sq_sum'][idx] / c_pre - pre_mean_soc**2) if c_pre > 0 else np.nan
                c_g = int(part['post_g_down_count'][idx])
                post_g_mean = float(part['post_g_down_sum'][idx] / c_g) if c_g > 0 else np.nan
                c_rho = int(part['post_density_count'][idx])
                post_density_mean = float(part['post_density_sum'][idx] / c_rho) if c_rho > 0 else np.nan
                row = {
                    'agent_id': int(part['agent_id'][idx]),
                    'x': int(part['x'][idx]),
                    'y': int(part['y'][idx]),
                    's': int(part['s'][idx]),
                    'phi': float(part['phi'][idx]),
                    'mu': float(part['mu'][idx]),
                    'local_density_before_move': float(part['dens_before'][idx]),
                    'local_density_after_move': float(part['dens_after'][idx]),
                    'step_when_decided': int(part['step_decided'][idx]),
                    'evacuation_time': int(part['evac_time'][idx]),
                    'decision_frame': int(part['decision_frame'][idx]),
                    'decision_h': float(part['decision_h'][idx]) if np.isfinite(part['decision_h'][idx]) else np.nan,
                    'evacuation_frame': int(part['evacuation_frame'][idx]),
                    'evacuation_h': float(part['evacuation_h'][idx]) if np.isfinite(part['evacuation_h'][idx]) else np.nan,
                    'I_soc_at_decision': float(part['I_soc_at_decision'][idx]) if np.isfinite(part['I_soc_at_decision'][idx]) else np.nan,
                    'H_at_decision': float(part['H_at_decision'][idx]) if np.isfinite(part['H_at_decision'][idx]) else np.nan,
                    'local_m_at_decision': float(part['local_m_at_decision'][idx]) if np.isfinite(part['local_m_at_decision'][idx]) else np.nan,
                    'n_neighbors_at_decision': int(part['n_neighbors_at_decision'][idx]),
                    'V_at_decision': float(part['V_at_decision'][idx]) if np.isfinite(part['V_at_decision'][idx]) else np.nan,
                    'density_at_decision': float(part['density_at_decision'][idx]) if np.isfinite(part['density_at_decision'][idx]) else np.nan,
                    'g_down_at_decision': int(part['g_down_at_decision'][idx]),
                    'post_g_down_mean': post_g_mean,
                    'post_mean_density': post_density_mean,
                    'post_no_progress_frames': int(part['post_no_progress_frames'][idx]),
                    'post_stagnated_frames': int(part['post_stagnated_frames'][idx]),
                    'pre_I_soc_mean': pre_mean_soc,
                    'pre_I_soc_var': pre_var_soc,
                    'pre_I_soc_max': float(part['pre_I_soc_max'][idx]) if c_pre > 0 else np.nan,
                    'pre_I_soc_min': float(part['pre_I_soc_min'][idx]) if c_pre > 0 else np.nan,
                    'pre_I_soc_count': c_pre,
                    'is_safe': 1 if V_field[int(part['y'][idx]), int(part['x'][idx])] == 0 else 0,
                }
                all_agent_rows.append(row)
        pd.DataFrame(all_agent_rows).to_csv(os.path.join(output_dir, f"agents_frame_{frame:03d}.csv"), index=False)
        np.save(os.path.join(output_dir, f"density_frame_{frame:03d}.npy"), global_density)

        flat_exit_events = [ev for sub in gathered_exit_events for ev in sub]
        if flat_exit_events:
            df_exit = pd.DataFrame(flat_exit_events, columns=['abs_step', 'frame', 'h_ext', 'agent_id', 'x', 'y'])
            exit_path = os.path.join(output_dir, 'exit_events.csv')
            header = not os.path.exists(exit_path)
            df_exit.to_csv(exit_path, mode='a', header=header, index=False)

        flat_trace = [r for sub in gathered_social_trace for r in sub]
        if flat_trace:
            df_trace = pd.DataFrame(flat_trace, columns=['agent_id', 'abs_step', 'frame', 'h_ext', 'I_soc', 'H_i', 's_before_update', 'x', 'y', 'tag'])
            trace_path = os.path.join(output_dir, 'social_trace_sample.csv')
            header = not os.path.exists(trace_path)
            df_trace.to_csv(trace_path, mode='a', header=header, index=False)

        rec = {
            'frame': frame,
            'h_ext': current_h_ext,
            'frame_start_step': frame_start_step,
            'frame_end_step': frame_end_step,
            'steps_used_this_frame': steps_used_this_frame,
            'm_dec_inst': m_dec_inst,
            'm_dec_cum': m_dec_cum,
            'm_evac_inst': m_evac_inst,
            'm_evac_cum': m_evac_cum,
            'A_dec_frame': A_dec_frame,
            'A_dec_per_step': A_dec_frame / steps_used_this_frame if steps_used_this_frame > 0 else np.nan,
            'J_out_frame': J_out_frame,
            'J_out_per_step': J_out_frame / steps_used_this_frame if steps_used_this_frame > 0 else np.nan,
            'Q_transport': Q_transport,
            'q_transport': q_transport,
            'rho_up': rho_up,
            'Q_up': Q_up,
        }
        history_log.append(rec)
        print(f"[Frame {frame:03d}/{NUM_FRAMES-1}] h_ext={current_h_ext:.3f}, m_dec_cum={m_dec_cum:.4f}, m_evac_cum={m_evac_cum:.4f}, Q={Q_transport}, A_dec={A_dec_frame}, J_out={J_out_frame}, steps={steps_used_this_frame}")

print(f"Time: {time.time() - start}s")
if rank == 0:
    pd.DataFrame(history_log).to_csv(os.path.join(output_dir, "macro_stats.csv"), index=False)
    pd.DataFrame({'frame': np.arange(NUM_FRAMES, dtype=int), 'h_ext': H_VALUES}).to_csv(os.path.join(output_dir, 'h_schedule.csv'), index=False)
    print("=== 全計算プロセスが完了しました ===")
