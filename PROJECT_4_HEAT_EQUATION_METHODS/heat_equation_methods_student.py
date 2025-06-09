#!/usr/bin/env python3
"""
学生模板：有限长细杆热传导方程多种数值方法比较
文件：heat_equation_methods_student.py
重要：函数名称必须与参考答案一致！

实现四种数值方法求解一维热传导方程：
1. 显式差分法 (FTCS)
2. 隐式差分法 (Backward Euler)
3. Crank-Nicolson 方法
4. scipy.solve_ivp 方法

问题：u_t = a²u_xx，边界条件 u(0,t) = u(l,t) = 0
初始条件：u(x,0) = 1 for 10 ≤ x ≤ 11, 0 elsewhere
参数：a² = 10, l = 20, t ∈ [0, 25]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
import time
from typing import Tuple, Dict, Optional
import warnings

# 物理参数
ALPHA = 10.0  # 热扩散系数 a²
L = 20.0      # 杆长
T_FINAL = 25.0  # 终止时间

def create_initial_condition(x: np.ndarray) -> np.ndarray:
    """
    创建初始条件：u(x,0) = 1 for 10 ≤ x ≤ 11, 0 elsewhere
    
    参数:
        x: 空间网格点
    返回:
        初始温度分布
    
    物理背景: 在细杆的 [10,11] 区间内初始温度为1，其他位置为0
    数值方法: 直接在网格点上设置初始值
    
    实现步骤:
    1. 创建零数组
    2. 找到满足条件的网格点
    3. 设置对应位置的值为1
    """
    # TODO: 创建与x同形状的零数组
    # TODO: 创建布尔掩码，标识 10 ≤ x ≤ 11 的位置
    # TODO: 在满足条件的位置设置值为1
    # TODO: 返回初始条件数组
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def solve_ftcs(nx: int, nt: int, total_time: float = T_FINAL, 
               alpha: float = ALPHA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用前向时间中心空间(FTCS)显式方法求解热传导方程
    
    参数:
        nx: 空间网格点数
        nt: 时间步数
        total_time: 总模拟时间
        alpha: 热扩散系数
    
    返回:
        元组 (x_grid, t_grid, solution_matrix)
    
    物理背景: FTCS是最直观的差分格式，使用前向差分近似时间导数，中心差分近似空间导数
    数值方法: u_i^{n+1} = u_i^n + r(u_{i+1}^n - 2u_i^n + u_{i-1}^n)
    稳定性条件: r = α*Δt/(Δx)² ≤ 0.5
    
    实现步骤:
    1. 创建空间和时间网格
    2. 计算稳定性参数r
    3. 检查稳定性条件
    4. 初始化解矩阵
    5. 时间步进循环
    6. 应用边界条件
    """
    # TODO: 创建空间网格 x = linspace(0, L, nx)
    # TODO: 创建时间网格 t = linspace(0, total_time, nt)
    # TODO: 计算网格间距 dx, dt
    # TODO: 计算稳定性参数 r = alpha * dt / dx²
    # TODO: 检查稳定性条件 r ≤ 0.5，如果违反则发出警告
    # TODO: 初始化解矩阵 u(nt, nx)
    # TODO: 设置初始条件 u[0, :] = create_initial_condition(x)
    # TODO: 时间步进循环：对每个时间步n
    #       - 更新内部点：u[n+1, 1:-1] = u[n, 1:-1] + r*(u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
    #       - 应用边界条件：u[n+1, 0] = u[n+1, -1] = 0
    # TODO: 返回 (x, t, u)
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def solve_backward_euler(nx: int, nt: int, total_time: float = T_FINAL,
                        alpha: float = ALPHA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用后向欧拉隐式方法求解热传导方程
    
    参数:
        nx: 空间网格点数
        nt: 时间步数
        total_time: 总模拟时间
        alpha: 热扩散系数
    
    返回:
        元组 (x_grid, t_grid, solution_matrix)
    
    物理背景: 后向欧拉方法使用后向差分近似时间导数，无条件稳定
    数值方法: (I - rA)u^{n+1} = u^n，其中A是二阶差分算子矩阵
    稳定性: 无条件稳定，可以使用较大时间步长
    
    实现步骤:
    1. 创建网格和计算参数
    2. 构建三对角系数矩阵
    3. 初始化解矩阵
    4. 时间步进：每步求解线性方程组
    5. 应用边界条件
    """
    # TODO: 创建空间和时间网格
    # TODO: 计算网格间距和稳定性参数
    # TODO: 确定内部点数量 n_interior = nx - 2
    # TODO: 构建三对角矩阵 A：对角线元素为 -(1+2r)，上下对角线为 r
    #       使用 scipy.sparse.diags 构建稀疏矩阵
    # TODO: 初始化解矩阵和设置初始条件
    # TODO: 时间步进循环：
    #       - 构建右端向量（仅内部点）
    #       - 求解线性方程组 Au = b
    #       - 更新解并应用边界条件
    # TODO: 返回结果
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def solve_crank_nicolson(nx: int, nt: int, total_time: float = T_FINAL,
                        alpha: float = ALPHA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用Crank-Nicolson方法求解热传导方程
    
    参数:
        nx: 空间网格点数
        nt: 时间步数
        total_time: 总模拟时间
        alpha: 热扩散系数
    
    返回:
        元组 (x_grid, t_grid, solution_matrix)
    
    物理背景: Crank-Nicolson是隐式和显式方法的平均，具有二阶时间精度
    数值方法: (I - r/2*A)u^{n+1} = (I + r/2*A)u^n
    精度特点: 时间精度O(Δt²)，空间精度O(Δx²)，无条件稳定
    
    实现步骤:
    1. 创建网格和计算参数
    2. 构建左端和右端系数矩阵
    3. 初始化解矩阵
    4. 时间步进：每步求解线性方程组
    5. 应用边界条件
    """
    # TODO: 创建空间和时间网格
    # TODO: 计算网格间距和稳定性参数
    # TODO: 确定内部点数量
    # TODO: 构建左端矩阵 A_left: (I - r/2*A)
    #       对角线: -(1+r), 上下对角线: r/2
    # TODO: 构建右端矩阵 A_right: (I + r/2*A)
    #       对角线: -(1-r), 上下对角线: -r/2
    # TODO: 初始化解矩阵和设置初始条件
    # TODO: 时间步进循环：
    #       - 计算右端向量: rhs = A_right @ u[n, 1:-1]
    #       - 求解线性方程组: A_left @ u_new = rhs
    #       - 更新解并应用边界条件
    # TODO: 返回结果
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def solve_with_scipy(nx: int, total_time: float = T_FINAL, alpha: float = ALPHA,
                    method: str = 'RK45', rtol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用scipy.solve_ivp通过将PDE转换为ODE系统来求解热传导方程
    
    参数:
        nx: 空间网格点数
        total_time: 总模拟时间
        alpha: 热扩散系数
        method: 积分方法 ('RK45', 'DOP853', 'Radau', 'BDF')
        rtol: 相对容差
    
    返回:
        元组 (x_grid, t_grid, solution_matrix)
    
    物理背景: 将PDE空间离散化后转换为ODE系统 du/dt = α*A*u
    数值方法: 使用高级ODE求解器，支持自适应步长
    优势: 自动步长控制，高精度，多种积分器可选
    
    实现步骤:
    1. 创建空间网格
    2. 构建空间差分算子矩阵
    3. 定义ODE系统右端函数
    4. 设置初始条件（仅内部点）
    5. 调用solve_ivp求解
    6. 重构完整解（包含边界条件）
    """
    # TODO: 创建空间网格
    # TODO: 计算空间步长 dx
    # TODO: 确定内部点数量
    # TODO: 构建空间差分算子矩阵 A_spatial
    #       使用二阶中心差分：[1, -2, 1]/(dx²)
    # TODO: 定义ODE系统函数 ode_system(t, u_interior)
    #       返回 alpha * A_spatial @ u_interior
    # TODO: 设置初始条件（仅内部点）
    # TODO: 定义时间范围 t_span = (0, total_time)
    # TODO: 调用solve_ivp求解ODE系统
    # TODO: 创建均匀时间网格用于输出
    # TODO: 在时间网格上评估解
    # TODO: 重构完整解矩阵（添加边界条件）
    # TODO: 返回结果
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def calculate_errors(u_numerical: np.ndarray, u_reference: np.ndarray, 
                    dx: float) -> Dict[str, float]:
    """
    计算数值解和参考解之间的各种误差范数
    
    参数:
        u_numerical: 数值解
        u_reference: 参考解
        dx: 空间网格间距
    
    返回:
        包含不同误差测量的字典
    
    物理背景: 误差分析是验证数值方法精度的重要手段
    数值方法: 计算L2范数、最大范数、相对误差等
    
    实现步骤:
    1. 检查解的形状一致性
    2. 计算误差矩阵
    3. 计算各种误差范数
    4. 返回误差字典
    """
    # TODO: 检查u_numerical和u_reference形状是否一致
    # TODO: 计算误差矩阵 error = u_numerical - u_reference
    # TODO: 计算L2误差: sqrt(sum(error²) * dx)
    # TODO: 计算L2范数的参考解: sqrt(sum(u_reference²) * dx)
    # TODO: 计算相对L2误差: l2_error / (l2_norm_ref + 1e-12)
    # TODO: 计算最大绝对误差: max(|error|)
    # TODO: 计算相对最大误差
    # TODO: 计算RMS误差: sqrt(mean(error²))
    # TODO: 返回误差字典
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def compare_methods(nx: int = 101, nt: int = 1000) -> Dict:
    """
    比较所有四种数值方法
    
    参数:
        nx: 空间网格点数
        nt: 时间步数
    
    返回:
        包含结果和计时信息的字典
    
    物理背景: 通过系统比较不同方法的精度和效率来选择最适合的求解器
    数值方法: 运行所有方法并计算相对误差和计算时间
    
    实现步骤:
    1. 运行所有四种方法并计时
    2. 选择参考解（通常是最高精度的方法）
    3. 计算各方法相对于参考解的误差
    4. 整理和输出比较结果
    """
    # TODO: 初始化结果字典
    # TODO: 运行FTCS方法并计时
    # TODO: 运行后向欧拉方法并计时
    # TODO: 运行Crank-Nicolson方法并计时
    # TODO: 运行scipy.solve_ivp方法并计时
    # TODO: 选择Crank-Nicolson作为参考解
    # TODO: 将scipy解插值到公共时间网格
    # TODO: 计算各方法相对于参考解的误差
    # TODO: 打印比较摘要
    # TODO: 返回结果字典
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def plot_comparison(results: Dict, save_plots: bool = True):
    """
    创建综合比较图表
    
    参数:
        results: 来自compare_methods的结果字典
        save_plots: 是否保存图表到文件
    
    物理背景: 可视化是理解数值方法特性的重要工具
    数值方法: 创建多种图表展示解的演化、误差分析、效率比较等
    
    实现步骤:
    1. 提取各方法的解
    2. 创建多子图布局
    3. 绘制不同时刻的温度分布
    4. 绘制误差随时间演化
    5. 绘制计算时间比较
    6. 绘制3D表面图
    7. 绘制精度-效率散点图
    8. 绘制稳定性分析
    """
    # TODO: 从results中提取各方法的解
    # TODO: 创建多子图布局 (2x3)
    # TODO: 子图1：不同时刻的温度分布比较
    # TODO: 子图2：误差随时间的演化
    # TODO: 子图3：计算时间比较（柱状图）
    # TODO: 子图4：3D温度演化表面图
    # TODO: 子图5：精度vs效率散点图
    # TODO: 子图6：FTCS稳定性分析
    # TODO: 调整布局并保存图表
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def convergence_study(nx_values: list = [21, 41, 81, 161], 
                     nt_factor: int = 10) -> Dict:
    """
    对不同网格分辨率进行收敛性研究
    
    参数:
        nx_values: 要测试的空间网格大小列表
        nt_factor: 确定nt = nt_factor * nx的因子
    
    返回:
        包含收敛性数据的字典
    
    物理背景: 收敛性研究验证数值方法的理论精度阶数
    数值方法: 系统地细化网格，观察误差的减小规律
    
    实现步骤:
    1. 使用最细网格计算参考解
    2. 对每个网格大小运行所有方法
    3. 计算相对于参考解的误差
    4. 记录计算时间
    5. 分析收敛阶数
    """
    # TODO: 初始化收敛性数据字典
    # TODO: 使用最细网格计算参考解
    # TODO: 对每个网格大小循环：
    #       - 运行三种方法并计时
    #       - 将参考解插值到当前网格
    #       - 计算误差
    #       - 存储结果
    # TODO: 返回收敛性数据
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def plot_convergence(convergence_data: Dict, save_plots: bool = True):
    """
    绘制收敛性研究结果
    
    参数:
        convergence_data: 来自convergence_study的数据
        save_plots: 是否保存图表
    
    物理背景: 收敛性图表显示数值方法的精度阶数
    数值方法: 对数坐标下的误差vs网格间距图
    
    实现步骤:
    1. 提取网格间距和误差数据
    2. 创建对数坐标图
    3. 绘制各方法的收敛曲线
    4. 添加理论收敛阶数参考线
    5. 绘制效率比较图
    """
    # TODO: 提取dx值和误差数据
    # TODO: 创建双子图布局
    # TODO: 子图1：收敛性图（loglog）
    #       - 绘制各方法的误差vs dx
    #       - 添加O(Δx)和O(Δx²)参考线
    # TODO: 子图2：效率图（loglog）
    #       - 绘制误差vs计算时间
    # TODO: 调整布局并保存
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

if __name__ == "__main__":
    print("热传导方程求解器 - 多种方法比较")
    print("=" * 60)
    print(f"问题参数:")
    print(f"  定义域: [0, {L}]")
    print(f"  时间: [0, {T_FINAL}]")
    print(f"  热扩散系数: α² = {ALPHA}")
    print(f"  初始条件: u(x,0) = 1 for 10 ≤ x ≤ 11, 0 elsewhere")
    print(f"  边界条件: u(0,t) = u({L},t) = 0")
    print()
    
    try:
        # 主要比较
        print("开始方法比较...")
        results = compare_methods(nx=101, nt=500)
        
        # 创建比较图表
        print("\n创建比较图表...")
        plot_comparison(results)
        
        # 收敛性研究
        print("\n开始收敛性研究...")
        conv_data = convergence_study(nx_values=[21, 41, 81], nt_factor=8)
        
        # 绘制收敛性结果
        print("\n创建收敛性图表...")
        plot_convergence(conv_data)
        
        print("\n分析完成！")
        print("\n主要发现:")
        print("1. Crank-Nicolson方法提供最佳精度")
        print("2. FTCS方法最快但有稳定性约束")
        print("3. 隐式方法无条件稳定")
        print("4. scipy.solve_ivp通过自适应步长提供良好精度")
        
    except NotImplementedError as e:
        print(f"\n需要实现的函数: {e}")
        print("请完成所有标记为TODO的部分")
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()