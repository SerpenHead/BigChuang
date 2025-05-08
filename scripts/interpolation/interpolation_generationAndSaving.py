from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from interpolation_utils import create_grid
import os
import json
class OzoneInterpolator:
    """用于对臭氧数据进行逆距离加权插值并保存为多个时间戳命名的.npy文件的类"""
    
    def __init__(self, region_path, grid_size=500, k=5, power=2):
        """
        初始化插值器，加载区域数据并设置参数。

        参数:
            region_path (str): 区域GeoJSON文件路径
            grid_size (float): 网格单元大小（单位：米，默认500）
            k (int): IDW插值使用的最近邻点数量（默认5）
            power (float): IDW插值距离权重的幂参数（默认2）
        """
        # 区域数据：GeoDataFrame，投影到EPSG:3857（米单位）
        self.region = gpd.read_file(region_path).to_crs(epsg=3857)
        # 网格大小：每个网格单元的边长（米）
        self.grid_size = grid_size
        # IDW最近邻点数量
        self.k = k
        # IDW距离权重幂参数
        self.power = power
        # 网格数据：GeoDataFrame，包含网格多边形
        self.grid = self._setup_grid()
        
    def _setup_grid(self):
        """创建网格并计算中心点。

        返回:
            GeoDataFrame: 包含网格多边形和中心点的地理数据框架
        """
        # 创建网格
        grid = create_grid(self.region, self.grid_size)
        # 添加中心点列
        grid["centroid"] = grid.centroid
        return grid
    
    def _idw_interpolation(self, station_coords, values, targets):
        """
        执行逆距离加权（IDW）插值。

        参数:
            station_coords (ndarray): 站点坐标（x, y）
            values (ndarray): 站点处的臭氧浓度值
            targets (ndarray): 需要插值的目标点坐标（x, y）

        返回:
            ndarray: 插值后的臭氧浓度值
        """
        # 动态调整k：取站点数量和默认k的最小值，避免索引越界
        k = min(self.k, len(station_coords))
        # 构建KDTree用于快速查找最近邻
        tree = cKDTree(station_coords)
        # 查询k个最近邻的距离和索引
        dists, idxs = tree.query(targets, k=k)
        # 计算权重：距离的倒数幂，添加小值避免除零
        weights = 1 / (dists ** self.power + 1e-12)
        # 归一化权重
        weights /= weights.sum(axis=1)[:, None]
        # 计算加权平均
        return (values[idxs] * weights).sum(axis=1)
    
    def flatten_data(self, nested_data):
        """
        将嵌套数据结构展平为平面字典格式。

        参数:
            nested_data (dict): 嵌套数据，格式为 {时间戳: {站点名称: [臭氧浓度, 经度, 纬度]}}
                               或 {时间戳: [{站点名称: [臭氧浓度, 经度, 纬度]}]}

        返回:
            dict: 平面字典，包含 name, timestamp, ozone, lon, lat 列表
        """
        names = []
        timestamps = []
        ozone_values = []
        lons = []
        lats = []
        
        for timestamp, stations in nested_data.items():
            # 处理 {站点名称: [臭氧浓度, 经度, 纬度]} 格式
            if isinstance(stations, dict):
                for station_name, values in stations.items():
                    if not isinstance(values, list) or len(values) != 3:
                        print(f"警告：站点 {station_name} 在时间戳 {timestamp} 的数据格式错误")
                        continue
                    names.append(station_name)
                    timestamps.append(timestamp)
                    ozone_values.append(values[0])  # 臭氧浓度
                    lons.append(values[1])         # 经度
                    lats.append(values[2])         # 纬度
            # 处理 [{站点名称: [臭氧浓度, 经度, 纬度]}] 格式
            elif isinstance(stations, list):
                for station_dict in stations:
                    for station_name, values in station_dict.items():
                        if not isinstance(values, list) or len(values) != 3:
                            print(f"警告：站点 {station_name} 在时间戳 {timestamp} 的数据格式错误")
                            continue
                        names.append(station_name)
                        timestamps.append(timestamp)
                        ozone_values.append(values[0])  # 臭氧浓度
                        lons.append(values[1])         # 经度
                        lats.append(values[2])         # 纬度
            else:
                print(f"警告：时间戳 {timestamp} 的数据格式不支持")
                continue
        
        return {
            "name": names,
            "timestamp": timestamps,
            "ozone": ozone_values,
            "lon": lons,
            "lat": lats
        }
    def interpolate_and_save(self, stations_data, output_dir="output", timestamps_path="timestamps.csv", visualize=False):
        """
        对每个时间戳的臭氧数据进行插值并保存为单独的.npy文件，文件名基于时间戳。

        参数:
            stations_data (dict): 站点数据，格式为 {时间戳: [{站点名称: [臭氧浓度, 经度, 纬度]}]}
                            或平面字典 {name, timestamp, ozone, lon, lat}
            output_dir (str): 保存.npy文件的目录
            timestamps_path (str): 保存时间戳的.csv文件路径

        返回:
            list: 保存的.npy文件路径列表
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
       
        # 展平嵌套数据
        df = pd.DataFrame(self.flatten_data(stations_data))
        
        # 确保时间戳为datetime格式
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # 创建站点GeoDataFrame，初始坐标系为EPSG:4326
        gdf_stations = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
        ).to_crs(self.region.crs)  # 投影到区域坐标系
        
        # 获取唯一时间戳
        timestamps = df["timestamp"].unique()
        # 获取实际网格数量
        n_grids = len(self.grid)
        # 保存的文件路径列表
        saved_files = []
        
        # 获取网格中心点坐标
        centroids = np.vstack([self.grid["centroid"].x, self.grid["centroid"].y]).T
        
        # 对每个时间戳进行插值
        for timestamp in timestamps:
            # 筛选当前时间戳的站点数据
            stations = gdf_stations[gdf_stations["timestamp"] == timestamp]
            if len(stations) < 1:
                print(f"时间戳 {timestamp} 无可用数据")
                continue
            
            # 提取站点坐标
            station_coords = np.vstack([stations.geometry.x, stations.geometry.y]).T
            # 提取臭氧浓度值
            ozone_values = stations["ozone"].values
            
            # 执行IDW插值
            self.grid["ozone"] = self._idw_interpolation(station_coords, ozone_values, centroids)
            
            # 保存为单独的.npy文件
            timestamp_str = timestamp.strftime("%Y-%m-%d-%H")
            output_path = os.path.join(output_dir, f"{timestamp_str}.npy")
            np.save(output_path, self.grid["ozone"].values)
            saved_files.append(output_path)
            if visualize:
                # 可视化插值结果
                self._visualize(timestamp_str)
        
        
        return saved_files
    
    def _visualize(self, timestamp):
        """
        可视化插值后的臭氧浓度网格。

        参数:
            timestamp (datetime): 用于图表标题的时间戳
        """
        # 设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
        plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
        
        # 创建绘图
        fig, ax = plt.subplots(figsize=(10, 10))
        # 定义颜色映射：从绿到黄到红
        cmap = mcolors.LinearSegmentedColormap.from_list("ozone_cmap", ["green", "yellow", "red"])
        # 臭氧浓度的最小值和最大值
        vmin, vmax = self.grid["ozone"].min(), self.grid["ozone"].max()
        
        # 绘制热图
        self.grid.plot(column="ozone", cmap=cmap, linewidth=0.1, edgecolor='black', ax=ax, legend=False)
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("臭氧浓度 (μg/m³)")
        
        # 设置标题和绘图样式
        ax.set_title(f"臭氧浓度热图 - {timestamp}", fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
# 示例用法
if __name__ == "__main__":
    # 示例嵌套数据
    # data = {
    #     "2025-01-01": 
    #         {"luohu": [40, 114.116872, 22.562334],
    #         "nigang": [92, 114.1014, 22.5715],
    #         "tongxinling": [88, 114.1063, 22.5545]}
    #     ,
    #     "2025-01-02": [
    #         {"luohu": [45, 114.116872, 22.562334],
    #         "nigang": [95, 114.1014, 22.5715],
    #         "tongxinling": [90, 114.1063, 22.5545]}
    #     ]
    # }
    # 将JSON文件转换为字典格式

    # data = json.load(open("F:\dachuang_network\data\JSON_data_for_interpolating.json", "r", encoding="utf-8"))
    # 处理相对路径
    script_dir = Path(__file__).resolve().parent
    # 找到项目根目录（上上级的上级：.../Bigchuang）
    project_root = script_dir.parent.parent
    # 拼出 data 路径
    data_dir = project_root.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)  # 若不存在则创建
    json_path = data_dir / "JSON_data_for_interpolating.json"

    data = json.load(open(json_path, "r", encoding="utf-8"))
    print(data)
    # 创建插值器实例
    interpolator = OzoneInterpolator(region_path="./scripts/interpolation/futian_luohu.json")
    # output_dir = "./data/interpolation_data"
    output_dir = data_dir / "interpolation_data"
    output_dir.mkdir(parents=True, exist_ok=True)  # 若不存在则创建
    # 执行插值并保存
    ozone_data = interpolator.interpolate_and_save()