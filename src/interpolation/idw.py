import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional

class IDWInterpolator:
    def __init__(self, power: float = 2):
        """
        初始化反距离加权插值器
        
        Args:
            power: IDW 的幂次参数，默认为2
        """
        self.power = power
        
    def create_grid(self, 
                    bounds: Tuple[float, float, float, float],
                    resolution: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建规则网格
        
        Args:
            bounds: 边界框 (min_lon, min_lat, max_lon, max_lat)
            resolution: 网格分辨率 (lon_res, lat_res)
            
        Returns:
            tuple: (经度网格, 纬度网格)
        """
        lon = np.arange(bounds[0], bounds[2], resolution[0])
        lat = np.arange(bounds[1], bounds[3], resolution[1])
        return np.meshgrid(lon, lat)
        
    def interpolate(self,
                   station_coords: np.ndarray,
                   station_values: np.ndarray,
                   grid_points: np.ndarray) -> np.ndarray:
        """
        执行IDW插值
        
        Args:
            station_coords: 站点坐标数组，形状为 (n_stations, 2)
            station_values: 站点观测值数组，形状为 (n_stations,)
            grid_points: 网格点坐标数组，形状为 (n_points, 2)
            
        Returns:
            np.ndarray: 插值结果数组
        """
        # 计算距离矩阵
        distances = cdist(grid_points, station_coords)
        
        # 处理距离为0的情况（网格点正好在站点位置）
        distances = np.where(distances == 0, np.finfo(float).eps, distances)
        
        # 计算权重
        weights = 1.0 / (distances ** self.power)
        weights_sum = np.sum(weights, axis=1)
        
        # 计算插值结果
        interpolated = np.sum(weights * station_values, axis=1) / weights_sum
        
        return interpolated

def main():
    # 示例使用
    # 创建一些模拟的站点数据
    stations = np.array([
        [116.3, 39.9],  # 北京
        [121.4, 31.2],  # 上海
        [113.2, 23.1],  # 广州
    ])
    
    values = np.array([120, 80, 100])  # 模拟的臭氧浓度值
    
    # 创建插值器
    interpolator = IDWInterpolator(power=2)
    
    # 定义网格范围和分辨率
    bounds = (110, 20, 125, 45)  # 覆盖中国东部地区
    resolution = (0.5, 0.5)  # 0.5度分辨率
    
    # 创建网格
    lon_grid, lat_grid = interpolator.create_grid(bounds, resolution)
    grid_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    
    # 执行插值
    result = interpolator.interpolate(stations, values, grid_points)
    
    # 重塑结果为二维数组
    result_grid = result.reshape(lon_grid.shape)
    
    print("插值完成！网格形状:", result_grid.shape)
    print(result)
    
if __name__ == "__main__":
    main() 