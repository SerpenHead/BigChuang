import geopandas as gpd
from shapely.geometry import box
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 投影为米单位坐标系
region = gpd.read_file("futian_luohu.json")
region = region.to_crs(epsg=3857)
def create_grid(gdf, grid_size):
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    xmin, ymin, xmax, ymax = bounds
    cols = np.arange(xmin, xmax, grid_size)
    rows = np.arange(ymin, ymax, grid_size)

    polygons = []
    for x in cols:
        for y in rows:
            poly = box(x, y, x + grid_size, y + grid_size)
            polygons.append(poly)

    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=gdf.crs)
    return grid[grid.intersects(gdf.union_all())]

grid_500 = create_grid(region, 500)

# 站点数据
data = {
    "name": ["luohu", "nigang", "tongxinling", "shennan", "lianhua", "shenzhen", "binhai", "OCT","minzhi","henggang","yantian"],
    "lat": [22.562334, 22.5715, 22.5545, 22.5465, 22.559, 22.543099, 22.5333, 22.542454, 22.6198, 22.64855, 22.59081],
    "lon": [114.116872, 114.1014, 114.1063, 114.0829, 114.0731, 114.057868, 114.0233, 113.987495, 114.0261, 114.2089, 114.2621],
    "ozone": [40, 92, 88, 50, 91, 20, 73, 10, 55, 24, 67]
}

df = pd.DataFrame(data)
gdf_stations = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
gdf_stations = gdf_stations.to_crs(region.crs)

# 获取网格中心点
grid_500["centroid"] = grid_500.centroid
centroids = np.vstack([grid_500["centroid"].x, grid_500["centroid"].y]).T

# 站点坐标
station_coords = np.vstack([gdf_stations.geometry.x, gdf_stations.geometry.y]).T
ozone_values = gdf_stations["ozone"].values

# 构建KDTree并进行 IDW 插值
tree = cKDTree(station_coords)

def idw_interpolation(tree, values, targets, k=5, power=2):
    dists, idxs = tree.query(targets, k=k)
    weights = 1 / (dists ** power + 1e-12)  # 避免除以0
    weights /= weights.sum(axis=1)[:, None]
    return (values[idxs] * weights).sum(axis=1)

grid_500["ozone"] = idw_interpolation(tree, ozone_values, centroids)

# 设置颜色映射：绿 → 黄 → 红
cmap = mcolors.LinearSegmentedColormap.from_list("ozone_cmap", ["green", "yellow", "red"])
vmin = grid_500["ozone"].min()
vmax = grid_500["ozone"].max()

fig, ax = plt.subplots(figsize=(10, 10))
grid_500.plot(column="ozone", cmap=cmap, linewidth=0.1, edgecolor='black', ax=ax, legend=False)
gdf_stations.plot(ax=ax, color='blue', markersize=50, marker='o', label='Ozone Stations', zorder=3)

# 站点名称
for x, y, label in zip(gdf_stations.geometry.x, gdf_stations.geometry.y, gdf_stations["name"]):
    ax.text(x, y, label, fontsize=9, ha='left', va='bottom', color='blue', zorder=4)

# 色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Ozone Concentration (μg/m³)")

ax.set_title("Ozone Concentration Heatmap with Monitoring Stations", fontsize=14)
ax.set_axis_off()
plt.tight_layout()
plt.legend()
plt.show()