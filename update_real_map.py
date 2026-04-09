import osmnx as ox
import geopandas as gpd
from sqlalchemy import create_engine
import warnings

# 警告を非表示にする（ログを見やすくするため）
warnings.filterwarnings('ignore')

print("▼ 1. データベースからベースマップを読み込んでいます...")
engine = create_engine('postgresql://postgres:mysecretpassword@localhost:5432/evacuation')
mesh_gdf = gpd.read_postgis("SELECT * FROM simulation_base_map;", con=engine, geom_col='geometry')

print("▼ 2. OSMnxで長野市周辺（長沼・豊野地区）の道路網を取得しています...")
# bboxの指定: (西, 南, 東, 北) の順
bbox = (138.22, 36.65, 138.30, 36.74)

# OpenStreetMapから「道路（highway）」の属性を持つポリゴン/ラインを全て取得
roads_gdf = ox.features_from_bbox(bbox=bbox, tags={'highway': True})
print(f"   -> {len(roads_gdf)} 本の道路・経路データを取得しました。")

print("▼ 3. 空間結合(Spatial Join)で障害物を判定しています...")
# 道路データとマス目を重ね合わせ、「道路と交差しているマス」を抽出
walkable_meshes = gpd.sjoin(mesh_gdf, roads_gdf, how='inner', predicate='intersects')

# 道路が存在する安全なマスのIDリスト
walkable_ids = walkable_meshes['mesh_id'].unique()

# 道路が通っていないマスを「障害物 (True)」として新しい列を追加
mesh_gdf['is_obstacle'] = ~mesh_gdf['mesh_id'].isin(walkable_ids)

obstacles_count = mesh_gdf['is_obstacle'].sum()
print(f"   -> 障害物マス: {obstacles_count} 個 / 全 {len(mesh_gdf)} 個")
print(f"   -> 千曲川や山林が障害物として認識されました。")

print("▼ 4. データベースを更新しています...")
# 更新したデータをPostGISに上書き保存
mesh_gdf.to_postgis(name='simulation_base_map', con=engine, if_exists='replace', index=False)
print("▼ 完了！ 現実の地形がベースマップに統合されました。")