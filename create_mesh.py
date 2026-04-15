import geopandas as gpd
import numpy as np
from shapely.geometry import box
from sqlalchemy import create_engine

print("▼ 1. データベースからハザードマップを読み込んでいます...")
engine = create_engine('postgresql://postgres:mysecretpassword@localhost:5432/evacuation')

# PostGISから先ほど保存したハザードマップを読み込む
hazard_gdf = gpd.read_postgis("SELECT * FROM hazard_map;", con=engine, geom_col='geometry')

print("▼ 2. シミュレーション用の空間メッシュ（マス目）を作成しています...")
# 範囲の定義
minx, miny, maxx, maxy = 138.22, 36.65, 138.30, 36.74

# メッシュのサイズ（0.001度は約100mに相当します）
grid_size = 0.0001

# x座標とy座標の配列を作成
x_coords = np.arange(minx, maxx, grid_size)
y_coords = np.arange(miny, maxy, grid_size)

# マス目（ポリゴン）のリストを作成
grid_polygons = []
for x in x_coords:
    for y in y_coords:
        # 1マスの四角形（Bounding Box）を作成
        grid_polygons.append(box(x, y, x + grid_size, y + grid_size))

# GeoDataFrameに変換
mesh_gdf = gpd.GeoDataFrame({'mesh_id': range(len(grid_polygons))}, geometry=grid_polygons, crs="EPSG:4326")
print(f"   -> {len(mesh_gdf)} 個のマス目（約10m四方）を作成しました。")

print("▼ 3. ハザードマップと重ね合わせて『危険度』を判定しています...")
# 空間結合（Spatial Join）：マス目がハザードマップと重なっているかを判定
# 浸水想定区域と重なるマス目だけを抽出
danger_mesh = gpd.sjoin(mesh_gdf, hazard_gdf, how="inner", predicate="intersects")

# 重複を排除して危険フラグを立てる
danger_mesh_ids = danger_mesh['mesh_id'].unique()

# 元のメッシュデータに「危険度（ポテンシャル V）」の列を追加
# 危険エリア=1, 安全エリア=0 とする
mesh_gdf['potential_v'] = mesh_gdf['mesh_id'].apply(lambda x: 1 if x in danger_mesh_ids else 0)

print(f"   -> 危険(V=1): {mesh_gdf['potential_v'].sum()} マス, 安全(V=0): {len(mesh_gdf) - mesh_gdf['potential_v'].sum()} マス")

print("▼ 4. PostGISへベースマップを保存しています...")
# シミュレーションの土台となるベースマップとして保存
mesh_gdf.to_postgis(name='simulation_base_map', con=engine, if_exists='replace', index=False)

print("▼ 完了！ データベースにシミュレーション用ベースマップが保存されました。")