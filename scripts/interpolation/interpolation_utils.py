import geopandas as gpd
import numpy as np
from shapely.geometry import box

def create_grid(gdf, grid_size):
    """
    在给定地理区域内创建方形网格。

    参数:
        gdf (GeoDataFrame): 需要创建网格的地理区域数据
        grid_size (float): 每个网格单元的大小（单位：米）

    返回:
        GeoDataFrame: 包含与区域相交的网格多边形的地理数据框架
    """
    # 获取区域边界 [minx, miny, maxx, maxy]
    bounds = gdf.total_bounds
    # 边界坐标：最小x、最小y、最大x、最大y
    xmin, ymin, xmax, ymax = bounds
    # 生成x方向的网格线
    cols = np.arange(xmin, xmax, grid_size)
    # 生成y方向的网格线
    rows = np.arange(ymin, ymax, grid_size)

    # 存储所有网格多边形
    polygons = []
    for x in cols:
        for y in rows:
            # 创建矩形网格单元
            poly = box(x, y, x + grid_size, y + grid_size)
            polygons.append(poly)

    # 创建包含网格几何的GeoDataFrame，继承区域的坐标系
    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=gdf.crs)
    # 仅保留与区域相交的网格
    return grid[grid.intersects(gdf.unary_union)]