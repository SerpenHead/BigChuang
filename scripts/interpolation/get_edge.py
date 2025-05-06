import osmnx as ox
import geopandas as gpd

# 查询福田区和罗湖区
places = ["Futian, Shenzhen, China", "Luohu, Shenzhen, China"]
gdf = ox.geocode_to_gdf(places)
# 保存为 GeoJSON
gdf.to_file("futian_luohu.json", driver="GeoJSON")