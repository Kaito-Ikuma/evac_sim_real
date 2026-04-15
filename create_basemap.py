from sqlalchemy import create_engine
import pandas as pd


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
df.to_csv("base_map_potential_real_10m.csv", index=False)
print("   -> 'base_map_potential_real.csv' として保存しました。")