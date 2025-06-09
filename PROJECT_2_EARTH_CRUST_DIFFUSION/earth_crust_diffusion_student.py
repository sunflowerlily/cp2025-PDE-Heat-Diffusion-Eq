#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地壳热扩散数值模拟 - 学生代码模板

本模块实现地壳中热传导方程的数值求解，考虑时变边界条件。
主要包括：
1. 隐式差分格式求解器
2. 长期演化分析
3. 季节性温度轮廓可视化
4. 解析解计算和比较
5. 参数敏感性分析

作者: [学生姓名]
学号: [学生学号]
日期: [完成日期]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import warnings
warnings.filterwarnings('ignore')

# 物理常数
D = 0.1  # 热扩散率 (m^2/day)
A = 10.0  # 年平均地表温度 (°C)
B = 12.0  # 地表温度振幅 (°C)
TAU = 365.0  # 年周期 (days)
T_BOTTOM = 11.0  # 20米深处温度 (°C)
T_INITIAL = 10.0  # 初始温度 (°C)
DEPTH_MAX = 20.0  # 最大深度 (m)


def validate_parameters(nz, nt, total_time_years):
    """
    验证输入参数的有效性
    
    参数:
        nz (int): 深度方向网格点数
        nt (int): 时间步数
        total_time_years (float): 总模拟时间（年）
    
    抛出:
        ValueError: 当参数无效时
    """
    if not isinstance(nz, int) or nz < 3:
        raise ValueError("nz must be an integer >= 3")
    if not isinstance(nt, int) or nt < 1:
        raise ValueError("nt must be an integer >= 1")
    if total_time_years <= 0:
        raise ValueError("total_time_years must be positive")


def solve_earth_crust_diffusion(nz=101, nt=3650, total_time_years=10.0):
    """
    求解地壳热扩散方程
    
    使用隐式差分格式求解一维热传导方程：
    ∂T/∂t = D * ∂²T/∂z²
    
    边界条件:
    - 地表 (z=0): T = A + B*sin(2πt/τ)
    - 深度20m: T = 11°C
    
    参数:
        nz (int): 深度方向网格点数，默认101
        nt (int): 时间步数，默认3650 (10年，每天一步)
        total_time_years (float): 总模拟时间（年），默认10.0
    
    返回:
        tuple: (depth_array, time_array, temperature_matrix)
            - depth_array (ndarray): 深度数组 (m)
            - time_array (ndarray): 时间数组 (days)
            - temperature_matrix (ndarray): 温度矩阵 [time, depth]
    
    TODO: 实现隐式差分格式求解
    提示:
    1. 设置空间和时间网格
    2. 构建三对角矩阵系统
    3. 在每个时间步处理边界条件
    4. 使用scipy.linalg.solve_banded求解
    """
    # TODO: 验证输入参数
    validate_parameters(nz, nt, total_time_years)
    
    # TODO: 设置网格
    # dz = DEPTH_MAX / (nz - 1)
    # dt = total_time_years * 365.0 / nt
    # depth = np.linspace(0, DEPTH_MAX, nz)
    # time = np.linspace(0, total_time_years * 365.0, nt + 1)
    
    # TODO: 初始化温度矩阵
    # temperature = np.zeros((nt + 1, nz))
    
    # TODO: 设置初始条件
    # temperature[0, :] = T_INITIAL
    # temperature[0, 0] = A + B * np.sin(2 * np.pi * 0 / TAU)  # 地表初始温度
    # temperature[0, -1] = T_BOTTOM  # 底部温度
    
    # TODO: 计算稳定性参数
    # eta = D * dt / (dz**2)
    # print(f"稳定性参数 η = {eta:.4f}")
    
    # TODO: 构建三对角矩阵的系数
    # 主对角线: 1 + 2*eta
    # 上下对角线: -eta
    
    # TODO: 时间步进循环
    # for n in range(nt):
    #     # 计算当前时间的地表温度
    #     t_current = time[n + 1]
    #     T_surface = A + B * np.sin(2 * np.pi * t_current / TAU)
    #     
    #     # 构建右端向量
    #     # 处理边界条件
    #     # 求解线性方程组
    #     # 更新温度分布
    
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请实现地壳热扩散求解器")
    
    return depth, time, temperature


def analyze_long_term_evolution(depth, time, temperature):
    """
    分析长期温度演化特征
    
    分析系统达到周期性稳态的过程，计算温度振幅衰减和相位延迟。
    
    参数:
        depth (ndarray): 深度数组 (m)
        time (ndarray): 时间数组 (days)
        temperature (ndarray): 温度矩阵 [time, depth]
    
    返回:
        dict: 包含以下分析结果的字典
            - 'amplitude_decay': 振幅随深度的衰减
            - 'phase_delay': 相位延迟随深度的变化
            - 'steady_state_time': 达到稳态的时间
            - 'energy_conservation': 能量守恒检查结果
    
    TODO: 实现长期演化分析
    提示:
    1. 分析最后几个周期的温度变化
    2. 使用FFT分析各深度的温度振幅
    3. 计算相位延迟
    4. 检查能量守恒
    """
    # TODO: 选择分析时间段（最后2-3个年周期）
    # analysis_start_time = ...
    
    # TODO: 计算各深度处的温度振幅
    # 使用最大值-最小值或FFT方法
    # amplitudes = []
    
    # TODO: 计算相位延迟
    # 找到各深度温度峰值的时间，计算相对于地表的延迟
    # phase_delays = []
    
    # TODO: 检查是否达到周期性稳态
    # 比较连续几个周期的温度分布
    # steady_state_time = ...
    
    # TODO: 能量守恒检查
    # 计算系统总热能的变化
    # energy_conservation = ...
    
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请实现长期演化分析")
    
    results = {
        'amplitude_decay': None,
        'phase_delay': None,
        'steady_state_time': None,
        'energy_conservation': None
    }
    
    return results


def plot_seasonal_profiles(depth, temperature_profiles, seasons, save_path=None):
    """
    绘制季节性温度轮廓
    
    在同一图中显示四个季节的温度随深度变化，展示季节性差异。
    
    参数:
        depth (ndarray): 深度数组 (m)
        temperature_profiles (ndarray): 四个季节的温度轮廓 [4, nz]
        seasons (list): 季节标签，如 ['Spring', 'Summer', 'Autumn', 'Winter']
        save_path (str, optional): 保存图片的路径
    
    返回:
        matplotlib.figure.Figure: 温度轮廓图对象
    
    TODO: 实现季节性温度轮廓绘制
    提示:
    1. 创建图形和坐标轴
    2. 为每个季节绘制温度-深度曲线
    3. 设置图例、标签和标题（使用英文）
    4. 美化图表外观
    """
    # TODO: 创建图形
    # fig, ax = plt.subplots(figsize=(10, 8))
    
    # TODO: 定义季节颜色
    # colors = ['green', 'red', 'orange', 'blue']
    
    # TODO: 绘制各季节的温度轮廓
    # for i, (season, color) in enumerate(zip(seasons, colors)):
    #     ax.plot(temperature_profiles[i], depth, 
    #             label=season, color=color, linewidth=2)
    
    # TODO: 设置坐标轴
    # ax.set_xlabel('Temperature (°C)', fontsize=12)
    # ax.set_ylabel('Depth (m)', fontsize=12)
    # ax.set_title('Seasonal Temperature Profiles in Earth Crust', fontsize=14)
    
    # TODO: 反转y轴（深度向下）
    # ax.invert_yaxis()
    
    # TODO: 添加网格和图例
    # ax.grid(True, alpha=0.3)
    # ax.legend(fontsize=11)
    
    # TODO: 美化图表
    # plt.tight_layout()
    
    # TODO: 保存图片（如果指定路径）
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请实现季节性温度轮廓绘制")
    
    return fig


def analytical_solution(depth, time, surface_temp_params):
    """
    计算半无限介质中的解析解
    
    对于半无限介质，地表温度为正弦变化时的解析解：
    T(z,t) = A + B * exp(-z/δ) * sin(2πt/τ - z/δ)
    其中 δ = sqrt(Dτ/π) 是特征深度
    
    参数:
        depth (ndarray): 深度数组 (m)
        time (ndarray): 时间数组 (days)
        surface_temp_params (dict): 地表温度参数
            - 'A': 平均温度
            - 'B': 振幅
            - 'tau': 周期
    
    返回:
        ndarray: 解析解温度矩阵 [time, depth]
    
    TODO: 实现解析解计算
    提示:
    1. 计算特征深度 δ
    2. 使用解析解公式计算温度分布
    3. 注意处理边界条件的差异
    """
    # TODO: 提取参数
    # A_param = surface_temp_params['A']
    # B_param = surface_temp_params['B']
    # tau = surface_temp_params['tau']
    
    # TODO: 计算特征深度
    # delta = np.sqrt(D * tau / np.pi)
    # print(f"特征深度 δ = {delta:.2f} m")
    
    # TODO: 创建网格
    # T_mesh, Z_mesh = np.meshgrid(time, depth, indexing='ij')
    
    # TODO: 计算解析解
    # T_analytical = (A_param + 
    #                B_param * np.exp(-Z_mesh / delta) * 
    #                np.sin(2 * np.pi * T_mesh / tau - Z_mesh / delta))
    
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请实现解析解计算")
    
    return T_analytical


def compare_with_analytical(depth, time, T_numerical, T_analytical):
    """
    比较数值解与解析解
    
    计算数值解与解析解的差异，绘制比较图表。
    
    参数:
        depth (ndarray): 深度数组
        time (ndarray): 时间数组
        T_numerical (ndarray): 数值解
        T_analytical (ndarray): 解析解
    
    返回:
        dict: 包含误差分析结果
    
    TODO: 实现数值解与解析解的比较
    """
    # TODO: 计算误差
    # error = T_numerical - T_analytical
    # rmse = np.sqrt(np.mean(error**2))
    # max_error = np.max(np.abs(error))
    
    # TODO: 绘制比较图
    # 可以绘制某个时刻的温度分布比较
    # 或者某个深度的时间序列比较
    
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请实现数值解与解析解的比较")
    
    return {'rmse': None, 'max_error': None}


def parameter_sensitivity_analysis():
    """
    参数敏感性分析
    
    分析热扩散率和地表温度振幅对结果的影响。
    
    返回:
        dict: 敏感性分析结果
    
    TODO: 实现参数敏感性分析
    提示:
    1. 改变热扩散率D的值
    2. 改变地表温度振幅B的值
    3. 比较不同参数下的温度分布
    4. 绘制敏感性图表
    """
    # TODO: 定义参数变化范围
    # D_values = [0.05, 0.1, 0.2]  # 不同的热扩散率
    # B_values = [6, 12, 18]  # 不同的温度振幅
    
    # TODO: 对每组参数运行模拟
    # 比较结果差异
    
    # TODO: 绘制敏感性图表
    
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请实现参数敏感性分析")
    
    return {}


def create_animation(depth, time, temperature, save_path=None):
    """
    创建温度演化动画
    
    制作显示温度随时间变化的动画。
    
    参数:
        depth (ndarray): 深度数组
        time (ndarray): 时间数组
        temperature (ndarray): 温度矩阵
        save_path (str, optional): 保存动画的路径
    
    TODO: 实现温度演化动画（可选任务）
    """
    # TODO: 使用matplotlib.animation创建动画
    # 显示温度随深度变化的动态过程
    
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请实现温度演化动画（可选）")


def main():
    """
    主函数：运行完整的地壳热扩散模拟
    
    执行所有分析步骤并生成结果图表。
    """
    print("=" * 60)
    print("地壳热扩散数值模拟")
    print("=" * 60)
    
    # 设置计算参数
    nz = 101  # 深度网格点数
    nt = 3650  # 时间步数（10年）
    total_time_years = 10.0
    
    print(f"网格设置: nz={nz}, nt={nt}")
    print(f"模拟时间: {total_time_years} 年")
    print(f"物理参数: D={D} m²/day, A={A}°C, B={B}°C")
    
    try:
        # 1. 求解数值解
        print("\n1. 求解热扩散方程...")
        depth, time, temperature = solve_earth_crust_diffusion(nz, nt, total_time_years)
        print("   数值求解完成")
        
        # 2. 长期演化分析
        print("\n2. 分析长期演化特征...")
        evolution_results = analyze_long_term_evolution(depth, time, temperature)
        print("   演化分析完成")
        
        # 3. 绘制季节性温度轮廓
        print("\n3. 绘制季节性温度轮廓...")
        # 选择第10年的四个时间点
        year_10_start = int(9 * 365)  # 第10年开始
        season_times = [year_10_start + i * 91 for i in range(4)]  # 每3个月一个点
        season_profiles = temperature[season_times, :]
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        
        fig = plot_seasonal_profiles(depth, season_profiles, seasons, 
                                   'seasonal_temperature_profiles.png')
        print("   季节性轮廓图已保存")
        
        # 4. 计算解析解并比较
        print("\n4. 计算解析解并比较...")
        surface_params = {'A': A, 'B': B, 'tau': TAU}
        T_analytical = analytical_solution(depth, time, surface_params)
        comparison_results = compare_with_analytical(depth, time, temperature, T_analytical)
        print(f"   RMSE误差: {comparison_results.get('rmse', 'N/A')}")
        
        # 5. 参数敏感性分析
        print("\n5. 参数敏感性分析...")
        sensitivity_results = parameter_sensitivity_analysis()
        print("   敏感性分析完成")
        
        print("\n=" * 60)
        print("所有分析完成！请查看生成的图表文件。")
        print("=" * 60)
        
    except NotImplementedError as e:
        print(f"\n错误: {e}")
        print("请完成相应函数的实现后再运行。")
    except Exception as e:
        print(f"\n运行错误: {e}")
        print("请检查代码实现和参数设置。")


if __name__ == "__main__":
    main()