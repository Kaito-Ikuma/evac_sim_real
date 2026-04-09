import numpy as np
from mpi4py import MPI
import time

# 1. MPIの初期化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # 自分のプロセス番号 (0 または 1)
size = comm.Get_size()  # 全プロセス数

# 安全装置
if size != 2:
    if rank == 0:
        print("エラー: このプログラムは mpirun -n 2 python mpi_test.py で実行してください。")
    MPI.Finalize()
    exit()

# 2. マップと担当エリアの設定 (100x100グリッド)
# 左右に縦割りで2分割します。
# Rank 0の担当: X座標 0 〜 49
# Rank 1の担当: X座標 50 〜 99
my_x_min = 0 if rank == 0 else 50
my_x_max = 49 if rank == 0 else 99

# 3. エージェントの初期配置
# テストのため、100人を左右50人ずつ「境界線のすぐそば」に配置します
np.random.seed(100 + rank) # 乱数シードをコアごとに分ける
my_agents = []
for _ in range(50):
    if rank == 0:
        # Rank 0 は境界のすぐ左側 (x=48, 49) に配置
        my_agents.append([np.random.randint(48, 50), np.random.randint(0, 100)])
    else:
        # Rank 1 は境界のすぐ右側 (x=50, 51) に配置
        my_agents.append([np.random.randint(50, 52), np.random.randint(0, 100)])

if rank == 0:
    print("=== MPI マイグレーション（境界越え）テスト開始 ===")

# 4. シミュレーションループ (15ステップ)
for step in range(1, 16):
    
    # [フェーズ1] 各自のエリア内でランダムに移動
    for i in range(len(my_agents)):
        dx = np.random.choice([-1, 0, 1])
        dy = np.random.choice([-1, 0, 1])
        my_agents[i][0] += dx
        my_agents[i][1] += dy

    # [フェーズ2] 担当エリア(境界)を越えたエージェントの仕分け
    agents_to_send = []
    agents_to_keep = []
    
    for agent in my_agents:
        x, y = agent[0], agent[1]
        # 自分のX座標の担当範囲外に出たら、送信リストへ
        if x < my_x_min or x > my_x_max:
            agents_to_send.append(agent)
        else:
            agents_to_keep.append(agent)
            
    my_agents = agents_to_keep # 手元には残った人だけ更新する

    # [フェーズ3] エージェントの引き継ぎ通信 (Migration)
    dest = 1 if rank == 0 else 0 # 送信相手（0なら1へ、1なら0へ）
    
    # 【最重要】sendrecv を使って、互いに「出ていく人」を渡し、「入ってくる人」を受け取る
    incoming_agents = comm.sendrecv(agents_to_send, dest=dest, source=dest)
    
    # 受け取ったエージェントを自分のリストに追加
    my_agents.extend(incoming_agents)

    # [フェーズ4] ターミナルへの出力
    # 出力が混ざらないように Barrier でタイミングを揃える
    comm.Barrier()
    
    print(f"[Step {step:02d}] Rank {rank} | 管轄人数: {len(my_agents):2d}人 | 転出: {len(agents_to_send):2d}人 | 転入: {len(incoming_agents):2d}人")
    
    comm.Barrier()
    if rank == 0:
        print("-" * 60)
        time.sleep(0.5) # ログを見やすくするためのウェイト