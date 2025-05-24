import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch_idw import TorchIDWInterpolator
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