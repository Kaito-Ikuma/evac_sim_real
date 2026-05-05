
import numpy as np
import pandas as pd
import skfmm
from mpi4py import MPI
import os
import time
import argparse
import json
from collections import deque

PRIORITY = "C_POSTMEAN"
ENABLE_B = True
ENABLE_C = True

start = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive theory -> coarse -> fine evacuation simulation. "
            "This parameterized version is intended for C_max sweeps."
        )
    )
    parser.add_argument('--N_agents', type=int, default=2000)
    parser.add_argument('--C_max', type=float, default=50.0)
    parser.add_argument('--mu_scale', type=float, default=1.0,
                        help='Scale factor for the baseline mu distribution. mu=1.0 means the current baseline.')
    parser.add_argument('--mu_mean', type=float, default=None,
                        help='If specified, overrides baseline_mu_mean * mu_scale.')
    parser.add_argument('--mu_std', type=float, default=None,
                        help='If specified, overrides baseline_mu_std * mu_scale.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_root', type=str, default='simulation_results_adaptive_sweep_cmax')
    parser.add_argument('--run_id', type=str, default=None,
                        help='If omitted, generated from N, C_max, mu_scale, seed.')
    parser.add_argument('--base_map_csv', type=str, default='base_map_potential_real_10m.csv')
    return parser.parse_args()


ARGS = parse_args()


def fmt_param(x):
    if isinstance(x, int):
        return str(x)
    if abs(float(x) - int(float(x))) < 1e-12:
        return str(int(float(x)))
    return (f"{float(x):.4g}").replace('.', 'p').replace('-', 'm')


BASE_MU_MEAN = 2.0
BASE_MU_STD = 0.1
RUN_ID = ARGS.run_id or f"N{ARGS.N_agents}_C{fmt_param(ARGS.C_max)}_mu{fmt_param(ARGS.mu_scale)}_seed{ARGS.seed}"
ROOT_OUTPUT_DIR_GLOBAL = os.path.join(ARGS.run_root, RUN_ID)

print("▼ 1. CSVからポテンシャル場 V(x,y) を抽出しています...")
df = pd.read_csv(ARGS.base_map_csv)
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

N_AGENTS = int(ARGS.N_agents)
T = 0.01
C_max = float(ARGS.C_max)
C_MAX = C_max
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
MU_SCALE = float(ARGS.mu_scale)
mu_mean = float(ARGS.mu_mean) if ARGS.mu_mean is not None else BASE_MU_MEAN * MU_SCALE
mu_std = float(ARGS.mu_std) if ARGS.mu_std is not None else BASE_MU_STD * MU_SCALE
phi_mean, phi_std = 1.0, 0.05
SEED = int(ARGS.seed)
HALO_MOVE = 1
HALO_GHOST = int(np.ceil(R_radius))
MAX_RELAX_STEPS = 5000
MIN_RELAX_STEPS = 200
STABLE_STEPS_REQUIRED = 20
EVENT_TOL = 5
SOC_TRACE_MAX_AGENTS = 128
SOC_TRACE_PRE_WINDOW = 64
UPSTREAM_MAX_V = 5.0

# Adaptive h-scan settings.
# The coarse scan range is not fixed in advance; it is centered on a
# mean-field spinodal estimate with a safety margin.
COARSE_SCAN_STEP = 0.05
COARSE_SCAN_MIN_H = 0.0
COARSE_SCAN_SAFETY_WIDTH = 0.35
COARSE_SCAN_FALLBACK_WIDTH = 0.75
FINE_SCAN_HALF_WIDTH = 0.12
FINE_SCAN_POST_WIDTH = 0.35
FINE_SCAN_STEP = 0.005



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


# H_VALUES is supplied stage-by-stage by the adaptive driver.


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


def record_upstream_arrival_if_needed(arrays, k: int, abs_step: int, frame: int, current_h_ext: float):
    """
    決断後、初めて upstream region に入った瞬間を保存する。
    t_i^arr を「決断済み agent が安全域直前の upstream mask に初到達した時刻」と定義する。
    """
    if arrays['step_decided'][k] < 0:
        return
    if arrays['arrival_step_upstream'][k] >= 0:
        return
    if arrays['evac_time'][k] >= 0:
        return
    ax = int(arrays['x'][k])
    ay = int(arrays['y'][k])
    if ax < 0 or ax >= GRID_W or ay < 0 or ay >= GRID_H:
        return
    if UPSTREAM_MASK[ay, ax]:
        arrays['arrival_step_upstream'][k] = abs_step
        arrays['arrival_frame_upstream'][k] = frame
        arrays['arrival_h_upstream'][k] = current_h_ext
        arrays['V_at_arrival_upstream'][k] = V_field[ay, ax]
        arrays['bottleneck_id'][k] = int(BOTTLENECK_ID_FIELD[ay, ax])


def build_upstream_mask(max_v: float = 5.0):
    return (V_field > 0) & (V_field <= max_v) & np.isfinite(V_field) & (~obstacles)


def build_bottleneck_id_field(upstream_mask: np.ndarray):
    """
    Upstream region の 4近傍連結成分を bottleneck_id としてラベル付けする。
    upstream_mask == True のセルだけに 0,1,2,... の ID を与え、それ以外は -1。
    """
    labels = -np.ones(upstream_mask.shape, dtype=np.int32)
    h, w = upstream_mask.shape
    bid = 0
    for yy in range(h):
        for xx in range(w):
            if not upstream_mask[yy, xx] or labels[yy, xx] >= 0:
                continue
            stack = [(xx, yy)]
            labels[yy, xx] = bid
            while stack:
                cx, cy = stack.pop()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    if upstream_mask[ny, nx] and labels[ny, nx] < 0:
                        labels[ny, nx] = bid
                        stack.append((nx, ny))
            bid += 1
    return labels, bid


UPSTREAM_MASK = build_upstream_mask(UPSTREAM_MAX_V)
BOTTLENECK_ID_FIELD, N_BOTTLENECKS = build_bottleneck_id_field(UPSTREAM_MASK)



def run_single_sweep(H_VALUES, output_dir, stage_name, seed=42):
    """Run one quasi-static sweep for the supplied h schedule."""
    NUM_FRAMES = len(H_VALUES)
    danger_y, danger_x = np.where((V_field > 0) & (V_field < np.inf))
    np.random.seed(seed)
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
    ARRIVAL_FIELDS = [
        'arrival_step_upstream', 'arrival_frame_upstream', 'arrival_h_upstream',
        'V_at_arrival_upstream', 'bottleneck_id'
    ]
    PAYLOAD_FIELDS = BASE_FIELDS + A_FIELDS + B_FIELDS + C_FIELDS + ARRIVAL_FIELDS
    FIELD_INDEX = {name: i for i, name in enumerate(PAYLOAD_FIELDS)}
    INT_FIELDS = {
        'agent_id', 'x', 'y', 's', 'step_decided', 'evac_time',
        'decision_frame', 'evacuation_frame', 'n_neighbors_at_decision',
        'g_down_at_decision', 'post_g_down_count', 'post_density_count',
        'post_no_progress_frames', 'post_stagnated_frames', 'pre_I_soc_count',
        'arrival_step_upstream', 'arrival_frame_upstream', 'bottleneck_id'
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
        'arrival_step_upstream': -np.ones(local_n, dtype=np.int64),
        'arrival_frame_upstream': -np.ones(local_n, dtype=np.int64),
        'arrival_h_upstream': np.full(local_n, np.nan),
        'V_at_arrival_upstream': np.full(local_n, np.nan),
        'bottleneck_id': -np.ones(local_n, dtype=np.int64),
    }

    density_local = np.zeros((local_h + 2 * HALO_MOVE, GRID_W), dtype=np.int32)
    for xi, yi in zip(arrays['x'], arrays['y']):
        density_local[y_to_local(int(yi)), int(xi)] += 1

    sampled_agent_ids = set(np.linspace(0, N_AGENTS - 1, SOC_TRACE_MAX_AGENTS, dtype=int).tolist())
    ring_buffers = {int(aid): deque(maxlen=SOC_TRACE_PRE_WINDOW) for aid in arrays['agent_id'] if int(aid) in sampled_agent_ids}
    social_trace_rows_local = []

    history_log = []
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"=== MPI並列計算を開始します（Priority {PRIORITY}, stage={stage_name}） ===")
        print(f"h-schedule: total {NUM_FRAMES} points, min={H_VALUES.min():.3f}, max={H_VALUES.max():.3f}")
        print(f"upstream cells={int(np.sum(UPSTREAM_MASK))}, n_bottlenecks={N_BOTTLENECKS}, UPSTREAM_MAX_V={UPSTREAM_MAX_V}")

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
                    record_upstream_arrival_if_needed(arrays, k, abs_step, frame, current_h_ext)
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
                        local_exit_events.append((abs_step, frame, current_h_ext, int(arrays['agent_id'][k]), cx, cy, int(arrays['bottleneck_id'][k])))
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

                record_upstream_arrival_if_needed(arrays, k, abs_step, frame, current_h_ext)

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
                    local_exit_events.append((abs_step, frame, current_h_ext, int(arrays['agent_id'][k]), ax, ay, int(arrays['bottleneck_id'][k])))

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
                        'arrival_step_upstream': int(part['arrival_step_upstream'][idx]),
                        'arrival_frame_upstream': int(part['arrival_frame_upstream'][idx]),
                        'arrival_h_upstream': float(part['arrival_h_upstream'][idx]) if np.isfinite(part['arrival_h_upstream'][idx]) else np.nan,
                        'V_at_arrival_upstream': float(part['V_at_arrival_upstream'][idx]) if np.isfinite(part['V_at_arrival_upstream'][idx]) else np.nan,
                        'bottleneck_id': int(part['bottleneck_id'][idx]),
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
                df_exit = pd.DataFrame(flat_exit_events, columns=['abs_step', 'frame', 'h_ext', 'agent_id', 'x', 'y', 'bottleneck_id'])
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
        np.save(os.path.join(output_dir, 'bottleneck_id_field.npy'), BOTTLENECK_ID_FIELD)
        np.save(os.path.join(output_dir, 'upstream_mask.npy'), UPSTREAM_MASK)
        print(f"=== stage {stage_name} が完了しました ===")
    if rank == 0:
        return pd.DataFrame(history_log)
    return None


def arange_unique_inclusive(start, stop, step):
    if stop < start:
        return np.array([], dtype=np.float64)
    n = int(np.floor((stop - start) / step + 1e-12))
    vals = start + step * np.arange(n + 1, dtype=np.float64)
    vals = vals[vals <= stop + 1e-12]
    return np.unique(np.round(vals, 12))


def estimate_theory_spinodal_logistic():
    """Rough mean-field estimate. It is used only to log a candidate, not to restrict the scan."""
    beta = 2.0 / T_dec
    B = mu_mean
    theta = phi_mean
    disc = 1.0 - 4.0 / (B * beta) if B * beta > 0 else -1.0
    if disc <= 0:
        return {'B_beta': B * beta, 'h_sp_low_p': np.nan, 'h_sp_high_p': np.nan,
                'h_center_guess': theta, 'has_mf_spinodal': False}
    root = np.sqrt(disc)
    p_low = 0.5 * (1.0 - root)
    p_high = 0.5 * (1.0 + root)
    def h_of_p(p):
        return theta + (1.0 / beta) * np.log(p / (1.0 - p)) - B * p
    h_low = h_of_p(p_low)
    h_high = h_of_p(p_high)
    return {'B_beta': B * beta, 'p_low': p_low, 'p_high': p_high,
            'h_sp_low_p': h_low, 'h_sp_high_p': h_high,
            'h_center_guess': float(np.nanmax([h_low, h_high])), 'has_mf_spinodal': True}


def choose_up_sweep_spinodal_from_theory(theory):
    """
    Choose the spinodal expected on the upward h sweep.

    In the logistic mean-field approximation, h_sp_low_p corresponds to
    the disappearance of the low-decision branch during an increasing-h scan.
    If it is unavailable, fall back to h_center_guess or phi_mean.
    """
    if theory is None:
        return float(phi_mean), "fallback_phi_mean"

    candidates = []
    h_low = theory.get('h_sp_low_p', np.nan)
    h_high = theory.get('h_sp_high_p', np.nan)
    h_center = theory.get('h_center_guess', np.nan)

    if np.isfinite(h_low):
        candidates.append((float(h_low), 'h_sp_low_p_up_sweep'))
    if np.isfinite(h_center):
        candidates.append((float(h_center), 'h_center_guess'))
    if np.isfinite(h_high):
        candidates.append((float(h_high), 'h_sp_high_p'))

    if not candidates:
        return float(phi_mean), "fallback_phi_mean"

    nonneg = [(h, label) for h, label in candidates if h >= COARSE_SCAN_MIN_H]
    if nonneg:
        h, label = max(nonneg, key=lambda z: z[0])
        return float(h), label

    h, label = max(candidates, key=lambda z: z[0])
    return float(max(COARSE_SCAN_MIN_H, h)), label + '_clipped_to_min'


def build_theory_guided_coarse_schedule(theory):
    """
    Build the coarse h schedule from the theoretical spinodal estimate.

    This replaces the previous fixed 0.00--1.50 coarse scan. The scan is
    centered around the estimated upward-sweep spinodal and padded by a
    safety width. If the mean-field theory predicts no spinodal, scan around
    the effective threshold phi_mean with a wider fallback width.
    """
    h_pred, source = choose_up_sweep_spinodal_from_theory(theory)
    has_spinodal = bool(theory.get('has_mf_spinodal', False)) if theory is not None else False
    half_width = COARSE_SCAN_SAFETY_WIDTH if has_spinodal else COARSE_SCAN_FALLBACK_WIDTH

    h_min = max(COARSE_SCAN_MIN_H, h_pred - half_width)
    h_max = max(h_min + COARSE_SCAN_STEP, h_pred + half_width)

    schedule = arange_unique_inclusive(h_min, h_max, COARSE_SCAN_STEP)
    schedule = np.unique(np.round(np.concatenate([schedule, np.array([h_pred])]), 12))
    schedule = schedule[schedule >= COARSE_SCAN_MIN_H - 1e-12]

    meta = {
        'h_pred_coarse_center': float(h_pred),
        'prediction_source': source,
        'has_mf_spinodal': has_spinodal,
        'coarse_h_min': float(schedule.min()) if len(schedule) else np.nan,
        'coarse_h_max': float(schedule.max()) if len(schedule) else np.nan,
        'coarse_step': COARSE_SCAN_STEP,
        'coarse_safety_width': float(half_width),
        'n_coarse_frames': int(len(schedule)),
    }
    return schedule, meta


def detect_transition_peak(macro_df):
    dfm = macro_df.copy()
    if len(dfm) == 0:
        return 1.0, 'fallback_empty'
    score_cols = []
    if 'A_dec_frame' in dfm.columns:
        a = dfm['A_dec_frame'].astype(float).to_numpy()
        if np.nanmax(a) > 0:
            score_cols.append(a / (np.nanmax(a) + 1e-12))
    if {'m_dec_cum', 'h_ext'}.issubset(dfm.columns):
        h = dfm['h_ext'].astype(float).to_numpy()
        m = dfm['m_dec_cum'].astype(float).to_numpy()
        dm = np.abs(np.gradient(m, h, edge_order=1)) if len(h) >= 2 else np.zeros_like(m)
        if np.nanmax(dm) > 0:
            score_cols.append(dm / (np.nanmax(dm) + 1e-12))
    if not score_cols:
        return float(dfm['h_ext'].median()), 'fallback_median'
    score = np.nanmean(np.vstack(score_cols), axis=0)
    idx = int(np.nanargmax(score))
    return float(dfm.iloc[idx]['h_ext']), 'A_dec_plus_dmdecdh'


def build_fine_schedule(h_star):
    h1 = max(COARSE_SCAN_MIN_H, h_star - FINE_SCAN_HALF_WIDTH)
    h2 = h_star + FINE_SCAN_HALF_WIDTH
    h3 = h_star + FINE_SCAN_POST_WIDTH
    parts = [
        # Coarse preconditioning branch. This preserves the upward-sweep
        # history before entering the fine window.
        arange_unique_inclusive(COARSE_SCAN_MIN_H, max(COARSE_SCAN_MIN_H, h1 - 0.03), COARSE_SCAN_STEP),
        arange_unique_inclusive(max(COARSE_SCAN_MIN_H, h1 - 0.02), h1, 0.02),
        arange_unique_inclusive(h1 + FINE_SCAN_STEP, h2, FINE_SCAN_STEP),
        arange_unique_inclusive(h2 + 0.02, h3, 0.02),
    ]
    vals = np.concatenate([p for p in parts if len(p) > 0])
    vals = vals[vals >= COARSE_SCAN_MIN_H - 1e-12]
    return np.unique(np.round(vals, 12))


if __name__ == '__main__':
    ROOT_OUTPUT_DIR = ROOT_OUTPUT_DIR_GLOBAL
    if rank == 0:
        os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)
        run_config = {
            'run_id': RUN_ID,
            'output_dir': os.path.abspath(ROOT_OUTPUT_DIR),
            'N_AGENTS': N_AGENTS,
            'C_MAX': C_MAX,
            'C_max': C_max,
            'MU_SCALE': MU_SCALE,
            'BASE_MU_MEAN': BASE_MU_MEAN,
            'BASE_MU_STD': BASE_MU_STD,
            'mu_mean': mu_mean,
            'mu_std': mu_std,
            'phi_mean': phi_mean,
            'phi_std': phi_std,
            'sigma_J': sigma_J,
            'T_dec': T_dec,
            'seed': SEED,
            'base_map_csv': ARGS.base_map_csv,
            'mpi_size': size,
            'coarse_scan_step': COARSE_SCAN_STEP,
            'coarse_scan_safety_width': COARSE_SCAN_SAFETY_WIDTH,
            'fine_scan_half_width': FINE_SCAN_HALF_WIDTH,
            'fine_scan_post_width': FINE_SCAN_POST_WIDTH,
            'fine_scan_step': FINE_SCAN_STEP,
        }
        with open(os.path.join(ROOT_OUTPUT_DIR, 'run_config.json'), 'w', encoding='utf-8') as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)
        theory = estimate_theory_spinodal_logistic()
        print('=== Theory prediction, rough logistic mean-field ===')
        print(theory)
    else:
        theory = None

    theory = comm.bcast(theory, root=0)
    H_COARSE, coarse_meta = build_theory_guided_coarse_schedule(theory)

    if rank == 0:
        pd.DataFrame([theory]).to_csv(os.path.join(ROOT_OUTPUT_DIR, 'theory_prediction_logistic_mf.csv'), index=False)
        pd.DataFrame([coarse_meta]).to_csv(os.path.join(ROOT_OUTPUT_DIR, 'theory_guided_coarse_scan_config.csv'), index=False)
        pd.DataFrame({'frame': np.arange(len(H_COARSE), dtype=int), 'h_ext': H_COARSE}).to_csv(os.path.join(ROOT_OUTPUT_DIR, 'theory_guided_coarse_h_schedule.csv'), index=False)
        print('=== Theory-guided coarse scan config ===')
        print(coarse_meta)

    coarse_df = run_single_sweep(H_COARSE, os.path.join(ROOT_OUTPUT_DIR, 'stage1_coarse'), 'stage1_coarse', seed=SEED)

    if rank == 0:
        h_star, method = detect_transition_peak(coarse_df)
        pd.DataFrame([{'h_star_coarse': h_star, 'method': method}]).to_csv(os.path.join(ROOT_OUTPUT_DIR, 'detected_transition_peak.csv'), index=False)
        print(f'=== Coarse transition candidate: h*={h_star:.6f}, method={method} ===')
    else:
        h_star = None
    h_star = comm.bcast(h_star, root=0)

    H_FINE_ADAPTIVE = build_fine_schedule(h_star)
    fine_df = run_single_sweep(H_FINE_ADAPTIVE, os.path.join(ROOT_OUTPUT_DIR, 'stage2_fine'), 'stage2_fine', seed=SEED)

    if rank == 0:
        def summarize_stage(stage_name, df_stage, h_pred):
            """
            Summarize each scan stage.

            Added diagnostics:
              - delta_h_pred = h_peak_A_dec - h_pred
              - chi_max_dmdec_dh = max |d m_dec_cum / d h|
              - h_peak_chi = h where chi is maximal
              - demand_excess_area_sum_pos_AminusJ = sum max(A_dec_frame - J_out_frame, 0)

            chi_max is evaluated with np.gradient on the actual h schedule, so it
            also works for non-uniform h spacing.
            """
            if df_stage is None or len(df_stage) == 0:
                return None

            out = {
                'stage': stage_name,
                'n_frames': int(len(df_stage)),
                'h_pred_theory': float(h_pred) if np.isfinite(h_pred) else np.nan,
                'h_min': float(df_stage['h_ext'].min()),
                'h_max': float(df_stage['h_ext'].max()),
            }

            # A_dec peak
            if 'A_dec_frame' in df_stage.columns and len(df_stage) > 0:
                idx_A = df_stage['A_dec_frame'].astype(float).idxmax()
                A_peak = float(df_stage.loc[idx_A, 'A_dec_frame'])
                h_peak_A = float(df_stage.loc[idx_A, 'h_ext'])
            else:
                A_peak = np.nan
                h_peak_A = np.nan

            # J_out peak
            if 'J_out_frame' in df_stage.columns and len(df_stage) > 0:
                idx_J = df_stage['J_out_frame'].astype(float).idxmax()
                J_peak = float(df_stage.loc[idx_J, 'J_out_frame'])
                h_peak_J = float(df_stage.loc[idx_J, 'h_ext'])
            else:
                J_peak = np.nan
                h_peak_J = np.nan

            # Q max
            if 'Q_transport' in df_stage.columns and len(df_stage) > 0:
                idx_Q = df_stage['Q_transport'].astype(float).idxmax()
                Q_max = float(df_stage.loc[idx_Q, 'Q_transport'])
                h_peak_Q = float(df_stage.loc[idx_Q, 'h_ext'])
            else:
                Q_max = np.nan
                h_peak_Q = np.nan

            # Sharpness chi = max |d m_dec_cum / d h|
            chi_max = np.nan
            h_peak_chi = np.nan
            if {'h_ext', 'm_dec_cum'}.issubset(df_stage.columns) and len(df_stage) >= 2:
                h_arr = df_stage['h_ext'].astype(float).to_numpy()
                m_arr = df_stage['m_dec_cum'].astype(float).to_numpy()
                if len(np.unique(h_arr)) >= 2:
                    chi_arr = np.abs(np.gradient(m_arr, h_arr, edge_order=1))
                    idx_chi = int(np.nanargmax(chi_arr))
                    chi_max = float(chi_arr[idx_chi])
                    h_peak_chi = float(h_arr[idx_chi])

            # Discrete demand excess diagnostic: sum over frames of [A_dec - J_out]_+.
            demand_excess_area = np.nan
            if {'A_dec_frame', 'J_out_frame'}.issubset(df_stage.columns):
                demand_excess_area = float(
                    np.nansum(
                        np.maximum(
                            df_stage['A_dec_frame'].astype(float).to_numpy()
                            - df_stage['J_out_frame'].astype(float).to_numpy(),
                            0.0,
                        )
                    )
                )

            Lambda = A_peak / (J_peak + 1e-12) if np.isfinite(A_peak) and np.isfinite(J_peak) else np.nan
            delta_h_pred = h_peak_A - h_pred if np.isfinite(h_peak_A) and np.isfinite(h_pred) else np.nan

            out.update({
                'h_peak_A_dec': h_peak_A,
                'delta_h_pred': delta_h_pred,
                'A_dec_peak': A_peak,
                'h_peak_J_out': h_peak_J,
                'J_out_peak': J_peak,
                'h_peak_Q': h_peak_Q,
                'Q_max': Q_max,
                'Q_max_over_N': Q_max / N_AGENTS if np.isfinite(Q_max) else np.nan,
                'Lambda_Adec_over_Jout': Lambda,
                'chi_max_dmdec_dh': chi_max,
                'h_peak_chi': h_peak_chi,
                'demand_excess_area_sum_pos_AminusJ': demand_excess_area,
            })
            return out

        h_pred_summary = float(coarse_meta.get('h_pred_coarse_center', np.nan))
        rows = []
        for stage_name, df_stage in [('coarse', coarse_df), ('fine', fine_df)]:
            row = summarize_stage(stage_name, df_stage, h_pred_summary)
            if row is not None:
                rows.append(row)

        summary = pd.DataFrame(rows)
        summary.to_csv(os.path.join(ROOT_OUTPUT_DIR, 'adaptive_scan_summary.csv'), index=False)

        # A compact one-line manifest is useful when collecting many parameter runs.
        run_manifest = {
            'run_id': RUN_ID,
            'output_dir': os.path.abspath(ROOT_OUTPUT_DIR),
            'N_AGENTS': N_AGENTS,
            'C_MAX': C_MAX,
            'C_max': C_max,
            'MU_SCALE': MU_SCALE,
            'mu_mean': mu_mean,
            'mu_std': mu_std,
            'seed': SEED,
            'mpi_size': size,
            'T_DEC': T_dec,
            'h_pred_theory': h_pred_summary,
            'coarse_h_min': coarse_meta.get('coarse_h_min', np.nan),
            'coarse_h_max': coarse_meta.get('coarse_h_max', np.nan),
            'coarse_step': coarse_meta.get('coarse_step', np.nan),
            'coarse_safety_width': coarse_meta.get('coarse_safety_width', np.nan),
            'h_star_coarse': h_star,
        }
        if len(summary) > 0 and 'fine' in set(summary['stage']):
            fine_row = summary.loc[summary['stage'] == 'fine'].iloc[0].to_dict()
            for k, v in fine_row.items():
                if k != 'stage':
                    run_manifest[f'fine_{k}'] = v
        pd.DataFrame([run_manifest]).to_csv(os.path.join(ROOT_OUTPUT_DIR, 'run_manifest.csv'), index=False)

        print('=== adaptive scan completed ===')
        print(summary)
