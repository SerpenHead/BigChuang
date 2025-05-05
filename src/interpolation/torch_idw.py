import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class TorchIDWInterpolator(nn.Module):
    def __init__(self, power: float = 2.0):
        """
        初始化基于PyTorch的反距离加权插值器
        
        Args:
            power: IDW的幂次参数，默认为2
        """
        super().__init__()
        self.power = power
        
    def create_grid(self, 
                   bounds: Tuple[float, float, float, float],
                   resolution: Tuple[float, float],
                   device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建规则网格
        
        Args:
            bounds: 边界框 (min_lon, min_lat, max_lon, max_lat)
            resolution: 网格分辨率 (lon_res, lat_res)
            device: torch设备
            
        Returns:
            tuple: (经度网格, 纬度网格)
        """
        lon = torch.arange(bounds[0], bounds[2], resolution[0], device=device)
        lat = torch.arange(bounds[1], bounds[3], resolution[1], device=device)
        return torch.meshgrid(lon, lat, indexing='ij')
    
    def compute_distances(self, grid_points: torch.Tensor, station_coords: torch.Tensor) -> torch.Tensor:
        """
        计算网格点到站点的距离矩阵
        
        Args:
            grid_points: 形状为 (batch_size, n_points, 2) 的网格点坐标
            station_coords: 形状为 (batch_size, n_stations, 2) 的站点坐标
            
        Returns:
            torch.Tensor: 形状为 (batch_size, n_points, n_stations) 的距离矩阵
        """
        # 扩展维度以进行广播
        grid_expanded = grid_points.unsqueeze(2)  # (batch, n_points, 1, 2)
        stations_expanded = station_coords.unsqueeze(1)  # (batch, 1, n_stations, 2)
        
        # 计算欧氏距离
        distances = torch.sqrt(torch.sum((grid_expanded - stations_expanded) ** 2, dim=-1))
        return distances
    
    def forward(self,
               station_coords: torch.Tensor,
               station_values: torch.Tensor,
               grid_points: torch.Tensor) -> torch.Tensor:
        """
        执行IDW插值
        
        Args:
            station_coords: 形状为 (batch_size, n_stations, 2) 的站点坐标
            station_values: 形状为 (batch_size, n_stations) 的站点观测值
            grid_points: 形状为 (batch_size, n_points, 2) 的网格点坐标
            
        Returns:
            torch.Tensor: 形状为 (batch_size, n_points) 的插值结果
        """
        # 计算距离矩阵
        distances = self.compute_distances(grid_points, station_coords)
        
        # 处理距离为0的情况
        distances = torch.where(distances == 0, 
                              torch.tensor(torch.finfo(torch.float32).eps, device=distances.device),
                              distances)
        
        # 计算权重
        weights = 1.0 / (distances ** self.power)
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        
        # 计算插值结果
        station_values_expanded = station_values.unsqueeze(1)  # (batch, 1, n_stations)
        interpolated = torch.sum(weights * station_values_expanded, dim=-1) / weights_sum.squeeze(-1)
        
        return interpolated

def main():
    # 示例使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建一些模拟的站点数据
    stations = torch.tensor([
        [
            [116.3, 39.9],  # 北京
            [121.4, 31.2],  # 上海
            [113.2, 23.1],  # 广州
        ]
    ], device=device)  # 添加batch维度
    
    values = torch.tensor([
        [120, 80, 100]
    ], device=device)  # 模拟的臭氧浓度值
    
    # 创建插值器
    interpolator = TorchIDWInterpolator(power=2).to(device)
    
    # 定义网格范围和分辨率
    bounds = (110, 20, 125, 45)  # 覆盖中国东部地区
    resolution = (0.5, 0.5)  # 0.5度分辨率
    
    # 创建网格
    lon_grid, lat_grid = interpolator.create_grid(bounds, resolution, device)
    grid_points = torch.stack([lon_grid.flatten(), lat_grid.flatten()], dim=-1)
    grid_points = grid_points.unsqueeze(0)  # 添加batch维度
    
    # 执行插值
    with torch.no_grad():
        result = interpolator(stations, values, grid_points)
    
    # 重塑结果为二维数组
    result_grid = result.reshape(lon_grid.shape)
    
    print("插值完成！网格形状:", result_grid.shape)
    print(result_grid)

if __name__ == "__main__":
    main()