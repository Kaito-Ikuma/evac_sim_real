import geopandas as gpd
from shapely.geometry import box
from sqlalchemy import create_engine

# ---------------------------------------------------------
# 1. データの読み込み
# 国土交通省のデータは文字コードが「Shift-JIS (cp932)」であることが多いため指定します
# ---------------------------------------------------------
# ※ ここを実際の .shp ファイルのパスに書き換えてください
file_path = "A31-12_20_GML/Shape/A31-12_20.shp"

print("▼ 1. Shapefileを読み込んでいます（少し時間がかかります）...")
gdf = gpd.read_file(file_path, encoding='cp932')

# ---------------------------------------------------------
# 2. 座標参照系（CRS）の統一
# 世界測地系（EPSG:4326 / GPS標準の緯度経度）に統一します
# ---------------------------------------------------------
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")
    print("   -> 座標系をEPSG:4326に変換しました。")

# ---------------------------------------------------------
# 3. 切り抜く範囲（長沼・豊野地区周辺）の定義
# 以前の議論で設定した範囲：東経 138.22〜138.30, 北緯 36.65〜36.74
# ---------------------------------------------------------
# box(minx, miny, maxx, maxy) の順で指定します (x=経度, y=緯度)
minx, miny, maxx, maxy = 138.22, 36.65, 138.30, 36.74
target_bbox = box(minx, miny, maxx, maxy)

# ---------------------------------------------------------
# 4. データの切り抜き（クリッピング）
# ---------------------------------------------------------
print("▼ 2. 指定した矩形範囲でデータを切り抜いています...")
clipped_gdf = gpd.clip(gdf, target_bbox)

print(f"   -> 切り抜き完了: {len(clipped_gdf)} 件のポリゴンを抽出しました。")

# ---------------------------------------------------------
# 5. PostGIS（データベース）への保存
# ---------------------------------------------------------
print("▼ 3. PostGISへデータを書き込んでいます...")
engine = create_engine('postgresql://postgres:mysecretpassword@localhost:5432/evacuation')

# テーブル名を 'hazard_map' として保存します
clipped_gdf.to_postgis(name='hazard_map', con=engine, if_exists='replace', index=False)

print("▼ 完了！ データベースにハザードマップが保存されました。")