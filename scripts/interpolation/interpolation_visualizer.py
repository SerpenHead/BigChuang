import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from interpolation_utils import create_grid

class OzoneVisualizer:
    """用于加载.npy格式的臭氧数据并生成热图可视化的类"""
    
    def __init__(self, region_path, grid_size=500):
        """
        初始化可视化器，加载区域数据并设置网格参数。

        参数:
            region_path (str): 区域GeoJSON文件路径
            grid_size (float): 网格单元大小（单位：米，默认500）
        """
        # 区域数据：GeoDataFrame，投影到EPSG:3857（米单位）
        self.region = gpd.read_file(region_path).to_crs(epsg=3857)
        # 网格大小：每个网格单元的边长（米）
        self.grid_size = grid_size
        # 网格数据：GeoDataFrame，包含网格多边形
        self.grid = create_grid(self.region, self.grid_size)
    def inspect_npy(self, file_path):
        """
        查看 .npy 文件的结构，包括形状、数据类型和示例值。

        参数:
            file_path (str): .npy 文件路径
        """
        try:
            # 加载 .npy 文件
            data = np.load(file_path)
            
            # 打印基本信息
            print("=== .npy 文件结构 ===")
            print(f"文件路径: {file_path}")
            print(f"数组形状: {data.shape}")
            print(f"数据类型: {data.dtype}")
            print(f"时间戳数量: {data.shape[0]}")
            print(f"网格数量: {data.shape[1]}")
            print(f"总元素数: {data.size}")
            
            # 打印示例值（前几个时间戳的前几个网格）
            print("\n=== 示例数据（前 2 个时间戳，前 5 个网格） ===")
            for i in range(min(2, data.shape[0])):
                print(f"时间戳索引 {i}: {data[i, :5]}")
            
            # 打印统计信息
            print("\n=== 数据统计 ===")
            print(f"最小值: {np.min(data):.2f}")
            print(f"最大值: {np.max(data):.2f}")
            print(f"平均值: {np.mean(data):.2f}")
            print(f"标准差: {np.std(data):.2f}")
            
        except FileNotFoundError:
            print(f"错误：文件 {file_path} 不存在")
        except Exception as e:
            print(f"错误：加载文件时发生错误 - {str(e)}")
    
    def visualize_npy(self, npy_path):
        """
        加载.npy文件并为每个时间戳生成臭氧浓度热图。

        参数:
            npy_path (str): 臭氧数据的.npy文件路径
            timestamps (list or None): 可视化用的时间戳列表（可选）
        """
        # 加载.npy文件，形状为(时间戳数量, 网格行数, 网格列数)
        self.inspect_npy(npy_path)
        ozone_data = np.load(npy_path)
        
       
        timestamp = npy_path.split("/")[-1].split(".")[0]
        
       
        # 展平二维网格数据，匹配网格数量
        ozone_values = ozone_data.flatten()
        # 将臭氧浓度值赋值到网格
        self.grid["ozone"] = ozone_values
        
        # 创建绘图
        fig, ax = plt.subplots(figsize=(10, 10))
        # 定义颜色映射：从绿到黄到红
        cmap = mcolors.LinearSegmentedColormap.from_list("ozone_cmap", ["green", "yellow", "red"])
        # 臭氧浓度的最小值和最大值
        vmin, vmax = self.grid["ozone"].min(), self.grid["ozone"].max()
        # 设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
        plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
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
    # 创建可视化器实例
    visualizer = OzoneVisualizer(region_path="./scripts/interpolation/futian_luohu.json")
    npy_path = "./data/interpolation_data/2025-01-01-00.npy"      
    # 加载并可视化.npy文件
    visualizer.visualize_npy(npy_path)