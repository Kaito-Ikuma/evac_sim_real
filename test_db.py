import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine

# ---------------------------------------------------------
# 1. データベースへの接続エンジンを作成（橋を架ける）
# フォーマット: postgresql://ユーザー名:パスワード@ホスト:ポート/データベース名
# ---------------------------------------------------------
engine = create_engine('postgresql://postgres:mysecretpassword@localhost:5432/evacuation')

# ---------------------------------------------------------
# 2. テスト用の空間データを作成する（Python側での処理）
# 例として、長野市周辺のダミーポイントを2つ（自宅と避難所）作ります
# ---------------------------------------------------------
data = {
    'agent_id': [1, 2],
    'location_type': ['Home (Danger)', 'Shelter (Safe)']
}
# 経度(x), 緯度(y) の順で座標を指定
geometry = [Point(138.22, 36.65), Point(138.25, 36.68)]

# GeoDataFrameに変換 (EPSG:4326 は一般的なGPSの緯度経度システムです)
gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

print("▼ 1. Python上で作成した空間データ")
print(gdf)

# ---------------------------------------------------------
# 3. データをPostGISに書き込む（Python → データベース）
# ---------------------------------------------------------
# 'test_agents' というテーブル名でデータベースに保存します
gdf.to_postgis(name='test_agents', con=engine, if_exists='replace', index=False)
print("\n▼ 2. PostGISへのデータ書き込み成功！")

# ---------------------------------------------------------
# 4. データをPostGISから読み込む（データベース → Python）
# ---------------------------------------------------------
# SQLを書いて、データベースから直接空間データとして引き出します
sql = "SELECT * FROM test_agents;"
loaded_gdf = gpd.read_postgis(sql, con=engine, geom_col='geometry')

print("\n▼ 3. PostGISからSQLで読み込んだデータ")
print(loaded_gdf)