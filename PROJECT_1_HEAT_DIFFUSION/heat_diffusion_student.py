#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学生模板：铝棒热传导方程显式差分法数值解
文件：heat_diffusion_student.py
重要：函数名称必须与参考答案一致！

本模块使用显式差分法求解一维热传导方程，包含：
1. 显式差分格式的完整实现
2. 稳定性条件的验证和分析
3. 解析解的计算和比较
4. 误差分析和收敛性研究
5. 可视化和动画生成

学生姓名：[请填写]
学号：[请填写]
完成日期：[请填写]
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('ignore')

# 设置绘图参数
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# 物理参数常量
K = 237.0      # 热导率 W/(m·K)
C = 900.0      # 比热容 J/(kg·K)
RHO = 2700.0   # 密度 kg/m³
ALPHA = K / (C * RHO)  # 热扩散系数 m²/s

class HeatDiffusionSolver:
    """
    热传导方程求解器类 - 显式差分法
    
    专注于显式差分格式的实现和稳定性分析
    """
    
    def __init__(self, L=1.0, nx=50, nt=1000, total_time=0.1):
        """
        初始化求解器参数
        
        参数:
            L (float): 铝棒长度 (m)
            nx (int): 空间网格点数
            nt (int): 时间步数
            total_time (float): 总模拟时间 (s)
        """
        self.L = L
        self.nx = nx
        self.nt = nt
        self.total_time = total_time
        
        # 网格参数
        self.dx = L / (nx - 1)
        self.dt = total_time / nt
        self.eta = ALPHA * self.dt / (self.dx ** 2)
        
        # 空间和时间网格
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, total_time, nt + 1)
        
        print(f"网格参数: dx={self.dx:.6f}, dt={self.dt:.6f}, η={self.eta:.6f}")
        if self.eta > 0.25:
            print(f"警告: η = {self.eta:.6f} > 0.25, 可能不稳定！")
        
    def initial_condition(self):
        """
        设置初始条件: T(x,0) = 100 K
        
        返回:
            numpy.ndarray: 初始温度分布
        """
        # TODO: 实现初始条件设置
        # 提示：创建一个长度为nx的数组，初始温度为100K
        # 注意边界条件：T(0,t) = T(L,t) = 0
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def explicit_finite_difference(self, dt=None, nt=None):
        """
        使用显式差分法求解热传导方程
        
        前向欧拉时间离散和中心差分空间离散：
        T[i][j+1] = T[i][j] + η*(T[i+1][j] - 2*T[i][j] + T[i-1][j])
        
        稳定性条件: η = α*Δt/(Δx)² ≤ 1/4
        
        参数:
            dt (float): 时间步长，如果提供则重新计算网格参数
            nt (int): 时间步数，如果提供则重新计算网格参数
        
        返回:
            tuple: (时间数组, 温度矩阵)
                - 时间数组: 形状 (nt+1,)
                - 温度矩阵: 形状 (nx, nt+1)
        
        物理背景:
            热传导方程描述了热量在固体中的传播过程。对于一维情况：
            ∂T/∂t = α * ∂²T/∂x²
            其中α是热扩散系数。
        
        数值方法:
            显式差分法使用前向差分离散时间导数，中心差分离散空间导数。
            优点：实现简单，计算效率高
            缺点：有条件稳定，需要满足CFL条件
        
        实现步骤:
        1. 如果提供了新的时间参数，重新计算网格参数
        2. 检查稳定性条件 η ≤ 0.25
        3. 初始化温度矩阵 T_matrix[i,j] = T(x_i, t_j)
        4. 设置初始条件
        5. 时间步进循环：
           - 更新内部节点 (i = 1 到 nx-2)
           - 应用边界条件 T[0] = T[-1] = 0
        6. 返回时间数组和温度矩阵
        """
        # TODO: 重新计算网格参数（如果提供了新的时间参数）
        # 提示：如果dt不为None，更新self.dt, self.nt, self.eta, self.t
        # 提示：如果nt不为None，更新self.nt, self.dt, self.eta, self.t
        
        # TODO: 检查稳定性条件
        # 提示：如果η > 0.25，打印警告信息
        
        # TODO: 初始化温度矩阵
        # 提示：创建形状为(nx, nt+1)的零矩阵
        # 提示：设置初始条件 T_matrix[:, 0] = self.initial_condition()
        
        # TODO: 时间步进循环
        # 提示：对于每个时间步j (0到nt-1):
        #   - 对于每个内部空间节点i (1到nx-2):
        #     T_matrix[i, j+1] = T_matrix[i, j] + eta * (T_matrix[i+1, j] - 2*T_matrix[i, j] + T_matrix[i-1, j])
        #   - 应用边界条件: T_matrix[0, j+1] = 0.0, T_matrix[-1, j+1] = 0.0
        
        # TODO: 返回结果
        # 提示：返回 (self.t, T_matrix)
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def analytical_solution(self, n_terms=50):
        """
        计算热传导方程的解析解
        
        解析解为傅里叶级数:
        T(x,t) = Σ(n=1,3,5,...) (4*T₀)/(n*π) * sin(n*π*x/L) * exp(-n²*π²*α*t/L²)
        其中 T₀ = 100 K (初始温度)
        
        参数:
            n_terms (int): 傅里叶级数项数
        
        返回:
            tuple: (时间数组, 温度矩阵)
        
        数学背景:
            对于初始条件T(x,0)=T₀，边界条件T(0,t)=T(L,t)=0的热传导方程，
            可以用分离变量法求得解析解。解为无穷级数形式，只包含奇数项。
        
        实现步骤:
        1. 初始化解析解矩阵
        2. 创建空间和时间网格矩阵
        3. 计算傅里叶级数（只考虑奇数项 n = 1,3,5,...）
        4. 应用边界条件
        5. 返回结果
        """
        # TODO: 初始化解析解矩阵
        # 提示：创建形状为(nx, nt+1)的零矩阵
        
        # TODO: 创建空间和时间网格矩阵
        # 提示：使用np.meshgrid(self.x, self.t, indexing='ij')
        
        # TODO: 计算傅里叶级数
        # 提示：对于n = 1,3,5,...,2*n_terms-1:
        #   coefficient = (4 * 100.0) / (n * np.pi)
        #   spatial_part = np.sin(n * np.pi * X / self.L)
        #   temporal_part = np.exp(-n**2 * np.pi**2 * ALPHA * T_time / self.L**2)
        #   T_analytical += coefficient * spatial_part * temporal_part
        
        # TODO: 应用边界条件
        # 提示：T_analytical[0, :] = 0.0, T_analytical[-1, :] = 0.0
        
        # TODO: 返回结果
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def plot_evolution(self, t_array, T_matrix, title="温度演化", save_fig=False):
        """
        绘制温度随时间的演化
        
        参数:
            t_array (np.ndarray): 时间数组
            T_matrix (np.ndarray): 温度矩阵
            title (str): 图表标题
            save_fig (bool): 是否保存图片
        """
        # TODO: 创建子图
        # 提示：使用plt.subplots(1, 2, figsize=(15, 6))
        
        # TODO: 绘制不同时间的温度分布
        # 提示：选择几个时间点，绘制T vs x的曲线
        
        # TODO: 绘制特定位置的温度演化
        # 提示：选择几个空间位置，绘制T vs t的曲线
        
        # TODO: 设置标签、标题、图例等
        
        # TODO: 保存图片（如果需要）
        
        # TODO: 显示图片
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def plot_3d_surface(self, t_array, T_matrix, title="3D温度分布", save_fig=False):
        """
        绘制3D温度分布图
        
        参数:
            t_array (np.ndarray): 时间数组
            T_matrix (np.ndarray): 温度矩阵
            title (str): 图表标题
            save_fig (bool): 是否保存图片
        """
        # TODO: 创建3D图形
        # 提示：使用fig.add_subplot(111, projection='3d')
        
        # TODO: 创建网格矩阵
        # 提示：使用np.meshgrid(self.x, t_array)
        
        # TODO: 绘制3D表面
        # 提示：使用ax.plot_surface()
        
        # TODO: 设置标签和标题
        
        # TODO: 添加颜色条
        
        # TODO: 保存和显示图片
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def create_animation(self, t_array, T_matrix, filename="heat_diffusion.gif"):
        """
        创建温度演化动画
        
        参数:
            t_array (np.ndarray): 时间数组
            T_matrix (np.ndarray): 温度矩阵
            filename (str): 动画文件名
        """
        # TODO: 创建图形和轴
        
        # TODO: 设置绘图参数
        
        # TODO: 定义动画更新函数
        
        # TODO: 创建动画
        # 提示：使用animation.FuncAnimation()
        
        # TODO: 保存动画
        
        # TODO: 显示动画
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def analyze_stability():
    """
    分析显式差分法的稳定性条件
    
    分析内容包括:
    1. 测试不同η值下的数值稳定性
    2. 演示不稳定现象
    3. 验证稳定性条件 η ≤ 1/4
    4. 可视化稳定性边界
    """
    # TODO: 实现稳定性分析
    # 提示：测试不同的η值，观察解的稳定性
    # 提示：绘制稳定和不稳定情况的对比图
    
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def calculate_errors():
    """
    计算数值解与解析解之间的误差
    
    误差分析包括:
    1. L2范数误差
    2. 最大误差
    3. 网格收敛性分析
    4. 误差随时间的演化
    """
    # TODO: 实现误差分析
    # 提示：计算不同网格下的误差
    # 提示：分析收敛阶
    # 提示：绘制误差演化图
    
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def compare_with_analytical():
    """
    数值解与解析解的详细比较
    
    比较内容包括:
    1. 不同时间的温度分布比较
    2. 特定位置的时间演化比较
    3. 误差分布可视化
    4. 收敛性验证
    """
    # TODO: 实现数值解与解析解的比较
    # 提示：在同一图中绘制数值解和解析解
    # 提示：计算和可视化误差分布
    
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def demonstrate_instability():
    """
    演示数值不稳定现象
    
    演示内容包括:
    1. 使用过大的时间步长
    2. 观察解的发散
    3. 分析不稳定的原因
    4. 可视化不稳定过程
    """
    # TODO: 实现不稳定性演示
    # 提示：使用η > 0.25的参数
    # 提示：观察解的振荡和发散
    # 提示：与稳定解进行对比
    
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

if __name__ == "__main__":
    """
    主程序：演示显式差分法求解热传导方程
    """
    print("=" * 60)
    print("铝棒热传导 - 显式差分法")
    print("=" * 60)
    
    # 创建求解器实例
    solver = HeatDiffusionSolver(L=1.0, nx=81, nt=1000, total_time=0.1)
    
    try:
        # 1. 显式差分法求解
        print("\n1. 运行显式差分法...")
        t_num, T_num = solver.explicit_finite_difference()
        print(f"数值解计算完成，最终最高温度: {np.max(T_num[:, -1]):.2f} K")
        
        # 2. 解析解计算
        print("\n2. 计算解析解...")
        t_ana, T_ana = solver.analytical_solution()
        print(f"解析解计算完成，最终最高温度: {np.max(T_ana[:, -1]):.2f} K")
        
        # 3. 误差分析
        print("\n3. 误差分析...")
        error_l2 = np.linalg.norm(T_num - T_ana) / np.linalg.norm(T_ana)
        error_max = np.max(np.abs(T_num - T_ana))
        print(f"L2相对误差: {error_l2:.6f}")
        print(f"最大绝对误差: {error_max:.6f} K")
        
        # 4. 可视化
        print("\n4. 生成可视化图表...")
        solver.plot_evolution(t_num, T_num, "显式差分法解")
        solver.plot_3d_surface(t_num, T_num, "3D温度分布")
        
        # 5. 稳定性分析
        print("\n5. 稳定性分析...")
        analyze_stability()
        
        # 6. 比较分析
        print("\n6. 数值解与解析解比较...")
        compare_with_analytical()
        
        # 7. 误差分析
        print("\n7. 详细误差分析...")
        calculate_errors()
        
        # 8. 不稳定性演示
        print("\n8. 不稳定性演示...")
        demonstrate_instability()
        
        # 9. 创建动画
        print("\n9. 创建动画...")
        solver.create_animation(t_num, T_num, "heat_diffusion_explicit.gif")
        
        print("\n" + "=" * 60)
        print("所有分析完成！请查看生成的图表和文件。")
        print("=" * 60)
        
    except NotImplementedError as e:
        print(f"\n需要实现的函数: {e}")
        print("请完成所有TODO标记的部分。")
    except Exception as e:
        print(f"\n执行过程中出现错误: {e}")
        print("请检查实现。")