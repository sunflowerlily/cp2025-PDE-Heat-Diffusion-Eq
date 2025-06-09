#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地壳热扩散数值模拟 - 参考答案

本模块实现地壳中热传导方程的数值求解，考虑时变边界条件。
主要包括：
1. 隐式差分格式求解器
2. 长期演化分析
3. 季节性温度轮廓可视化
4. 解析解计算和比较
5. 参数敏感性分析

作者: 教学团队
日期: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.fft import fft, fftfreq
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
    """
    # 验证输入参数
    validate_parameters(nz, nt, total_time_years)
    
    # 设置网格
    dz = DEPTH_MAX / (nz - 1)
    dt = total_time_years * 365.0 / nt
    depth = np.linspace(0, DEPTH_MAX, nz)
    time = np.linspace(0, total_time_years * 365.0, nt + 1)
    
    # 初始化温度矩阵
    temperature = np.zeros((nt + 1, nz))
    
    # 设置初始条件
    temperature[0, :] = T_INITIAL
    temperature[0, 0] = A + B * np.sin(2 * np.pi * 0 / TAU)  # 地表初始温度
    temperature[0, -1] = T_BOTTOM  # 底部温度
    
    # 计算稳定性参数
    eta = D * dt / (dz**2)
    print(f"稳定性参数 η = {eta:.4f}")
    print(f"网格间距: dz = {dz:.3f} m, dt = {dt:.3f} days")
    
    # 构建三对角矩阵的系数
    # 对于隐式格式: (1 + 2η)T[i]^{n+1} - ηT[i-1]^{n+1} - ηT[i+1]^{n+1} = T[i]^n
    main_diag = 1 + 2 * eta  # 主对角线
    off_diag = -eta  # 上下对角线
    
    # 构建带状矩阵格式 (scipy.linalg.solve_banded)
    # ab[0, :] 是上对角线, ab[1, :] 是主对角线, ab[2, :] 是下对角线
    ab = np.zeros((3, nz))
    ab[0, 1:] = off_diag  # 上对角线
    ab[1, :] = main_diag  # 主对角线
    ab[2, :-1] = off_diag  # 下对角线
    
    # 时间步进循环
    for n in range(nt):
        # 计算当前时间的地表温度
        t_current = time[n + 1]
        T_surface = A + B * np.sin(2 * np.pi * t_current / TAU)
        
        # 构建右端向量
        rhs = temperature[n, :].copy()
        
        # 处理边界条件
        # 地表边界 (z=0): 固定温度
        ab[1, 0] = 1.0  # 主对角线第一个元素
        ab[0, 1] = 0.0  # 上对角线第一个元素
        rhs[0] = T_surface
        
        # 底部边界 (z=20m): 固定温度
        ab[1, -1] = 1.0  # 主对角线最后一个元素
        ab[2, -2] = 0.0  # 下对角线最后一个元素
        rhs[-1] = T_BOTTOM
        
        # 求解线性方程组
        temperature[n + 1, :] = solve_banded((1, 1), ab, rhs)
        
        # 恢复矩阵系数（为下一次迭代准备）
        ab[1, 0] = main_diag
        ab[0, 1] = off_diag
        ab[1, -1] = main_diag
        ab[2, -2] = off_diag
        
        # 进度显示
        if (n + 1) % 365 == 0:
            year = (n + 1) // 365
            print(f"  完成第 {year} 年模拟")
    
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
    """
    nz = len(depth)
    nt = len(time)
    
    # 选择分析时间段（最后2个年周期）
    analysis_start_idx = max(0, nt - 2 * 365)
    analysis_time = time[analysis_start_idx:]
    analysis_temp = temperature[analysis_start_idx:, :]
    
    # 计算各深度处的温度振幅
    amplitudes = np.zeros(nz)
    phase_delays = np.zeros(nz)
    
    for i in range(nz):
        temp_series = analysis_temp[:, i]
        
        # 方法1: 使用最大值-最小值计算振幅
        amplitudes[i] = (np.max(temp_series) - np.min(temp_series)) / 2
        
        # 方法2: 使用FFT分析主频率的振幅
        if len(temp_series) > 365:  # 确保有足够的数据点
            fft_result = fft(temp_series - np.mean(temp_series))
            freqs = fftfreq(len(temp_series), d=1.0)  # 假设每天一个数据点
            
            # 找到年频率对应的索引
            target_freq = 1.0 / 365.0  # 年频率
            freq_idx = np.argmin(np.abs(freqs - target_freq))
            
            # 更新振幅（FFT方法）
            amplitudes[i] = np.abs(fft_result[freq_idx]) / len(temp_series) * 2
            
            # 计算相位延迟
            phase = np.angle(fft_result[freq_idx])
            phase_delays[i] = -phase / (2 * np.pi) * 365.0  # 转换为天数
    
    # 检查是否达到周期性稳态
    # 比较最后两个年周期的温度分布
    if nt > 2 * 365:
        last_year = temperature[-365:, :]
        prev_year = temperature[-2*365:-365, :]
        
        # 计算相对误差
        relative_error = np.mean(np.abs(last_year - prev_year) / (np.abs(prev_year) + 1e-10))
        steady_state_reached = relative_error < 0.01  # 1%的阈值
        
        if steady_state_reached:
            steady_state_time = time[-2*365] / 365.0  # 转换为年
        else:
            steady_state_time = None
    else:
        steady_state_time = None
    
    # 能量守恒检查
    # 计算系统总热能的变化率
    dz = depth[1] - depth[0] if len(depth) > 1 else 1.0
    total_energy = np.trapz(temperature, dx=dz, axis=1)  # 沿深度积分
    
    # 计算能量变化率
    if len(total_energy) > 1:
        energy_change_rate = np.gradient(total_energy, time)
        max_energy_change_rate = np.max(np.abs(energy_change_rate))
        energy_conservation_good = max_energy_change_rate < 0.1  # 阈值
    else:
        energy_conservation_good = True
        max_energy_change_rate = 0.0
    
    results = {
        'amplitude_decay': amplitudes,
        'phase_delay': phase_delays,
        'steady_state_time': steady_state_time,
        'energy_conservation': {
            'is_conserved': energy_conservation_good,
            'max_change_rate': max_energy_change_rate
        }
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
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 定义季节颜色
    colors = ['green', 'red', 'orange', 'blue']
    markers = ['o', 's', '^', 'D']
    
    # 绘制各季节的温度轮廓
    for i, (season, color, marker) in enumerate(zip(seasons, colors, markers)):
        ax.plot(temperature_profiles[i], depth, 
                label=season, color=color, linewidth=2.5,
                marker=marker, markersize=4, markevery=5)
    
    # 设置坐标轴
    ax.set_xlabel('Temperature (°C)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')
    ax.set_title('Seasonal Temperature Profiles in Earth Crust\n(Year 10 Simulation)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 反转y轴（深度向下）
    ax.invert_yaxis()
    
    # 添加网格和图例
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    
    # 设置坐标轴范围
    temp_min = np.min(temperature_profiles) - 1
    temp_max = np.max(temperature_profiles) + 1
    ax.set_xlim(temp_min, temp_max)
    ax.set_ylim(depth[-1], depth[0])
    
    # 美化图表
    ax.tick_params(axis='both', which='major', labelsize=11)
    plt.tight_layout()
    
    # 添加文本注释
    ax.text(0.02, 0.98, f'Surface temp. range: {temp_min:.1f}°C to {temp_max:.1f}°C',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 保存图片（如果指定路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  季节性温度轮廓图已保存至: {save_path}")
    
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
    """
    # 提取参数
    A_param = surface_temp_params['A']
    B_param = surface_temp_params['B']
    tau = surface_temp_params['tau']
    
    # 计算特征深度
    delta = np.sqrt(D * tau / np.pi)
    print(f"特征深度 δ = {delta:.2f} m")
    
    # 创建网格
    T_mesh, Z_mesh = np.meshgrid(time, depth, indexing='ij')
    
    # 计算解析解
    T_analytical = (A_param + 
                   B_param * np.exp(-Z_mesh / delta) * 
                   np.sin(2 * np.pi * T_mesh / tau - Z_mesh / delta))
    
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
    """
    # 计算误差
    error = T_numerical - T_analytical
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    mean_abs_error = np.mean(np.abs(error))
    
    print(f"  数值解与解析解比较:")
    print(f"    RMSE: {rmse:.4f} °C")
    print(f"    最大绝对误差: {max_error:.4f} °C")
    print(f"    平均绝对误差: {mean_abs_error:.4f} °C")
    
    # 绘制比较图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 某个时刻的温度分布比较（第10年春季）
    time_idx = len(time) - 365  # 第10年开始
    ax1.plot(T_numerical[time_idx, :], depth, 'b-', linewidth=2, label='Numerical')
    ax1.plot(T_analytical[time_idx, :], depth, 'r--', linewidth=2, label='Analytical')
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_title('Temperature Profile Comparison\n(Year 10, Day 1)', fontsize=13)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 地表温度时间序列比较
    time_years = time / 365.0
    ax2.plot(time_years, T_numerical[:, 0], 'b-', linewidth=2, label='Numerical')
    ax2.plot(time_years, T_analytical[:, 0], 'r--', linewidth=2, label='Analytical')
    ax2.set_xlabel('Time (years)', fontsize=12)
    ax2.set_ylabel('Surface Temperature (°C)', fontsize=12)
    ax2.set_title('Surface Temperature Time Series', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 某个深度的温度时间序列比较（5米深度）
    depth_idx = len(depth) // 4  # 约5米深度
    ax3.plot(time_years, T_numerical[:, depth_idx], 'b-', linewidth=2, label='Numerical')
    ax3.plot(time_years, T_analytical[:, depth_idx], 'r--', linewidth=2, label='Analytical')
    ax3.set_xlabel('Time (years)', fontsize=12)
    ax3.set_ylabel('Temperature (°C)', fontsize=12)
    ax3.set_title(f'Temperature at {depth[depth_idx]:.1f}m Depth', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 误差分布
    im = ax4.contourf(time_years, depth, error.T, levels=20, cmap='RdBu_r')
    ax4.set_xlabel('Time (years)', fontsize=12)
    ax4.set_ylabel('Depth (m)', fontsize=12)
    ax4.set_title('Temperature Error (Numerical - Analytical)', fontsize=13)
    ax4.invert_yaxis()
    plt.colorbar(im, ax=ax4, label='Error (°C)')
    
    plt.tight_layout()
    plt.savefig('numerical_analytical_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  比较图已保存至: numerical_analytical_comparison.png")
    
    return {
        'rmse': rmse,
        'max_error': max_error,
        'mean_abs_error': mean_abs_error,
        'comparison_figure': fig
    }


def parameter_sensitivity_analysis():
    """
    参数敏感性分析
    
    分析热扩散率和地表温度振幅对结果的影响。
    
    返回:
        dict: 敏感性分析结果
    """
    print("\n  进行参数敏感性分析...")
    
    # 定义参数变化范围
    D_values = [0.05, 0.1, 0.2]  # 不同的热扩散率
    B_values = [6, 12, 18]  # 不同的温度振幅
    
    # 存储结果
    results = {
        'D_sensitivity': {},
        'B_sensitivity': {}
    }
    
    # 基准参数
    base_nz, base_nt = 51, 1825  # 减少计算量
    base_time_years = 5.0
    
    # 1. 热扩散率敏感性分析
    print("    分析热扩散率敏感性...")
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    global D  # 临时修改全局变量
    original_D = D
    
    for i, D_val in enumerate(D_values):
        D = D_val
        depth, time, temperature = solve_earth_crust_diffusion(base_nz, base_nt, base_time_years)
        
        # 绘制最终温度分布
        ax1.plot(temperature[-1, :], depth, linewidth=2, 
                label=f'D = {D_val} m²/day')
        
        # 绘制地表温度演化
        ax2.plot(time/365.0, temperature[:, 0], linewidth=2,
                label=f'D = {D_val} m²/day')
        
        results['D_sensitivity'][D_val] = {
            'final_profile': temperature[-1, :],
            'surface_evolution': temperature[:, 0]
        }
    
    D = original_D  # 恢复原值
    
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_title('Final Temperature Profiles\n(Different Thermal Diffusivity)', fontsize=13)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Time (years)', fontsize=12)
    ax2.set_ylabel('Surface Temperature (°C)', fontsize=12)
    ax2.set_title('Surface Temperature Evolution\n(Different Thermal Diffusivity)', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('thermal_diffusivity_sensitivity.png', dpi=300, bbox_inches='tight')
    
    # 2. 地表温度振幅敏感性分析
    print("    分析地表温度振幅敏感性...")
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
    
    global B  # 临时修改全局变量
    original_B = B
    
    for i, B_val in enumerate(B_values):
        B = B_val
        depth, time, temperature = solve_earth_crust_diffusion(base_nz, base_nt, base_time_years)
        
        # 绘制最终温度分布
        ax3.plot(temperature[-1, :], depth, linewidth=2,
                label=f'B = {B_val} °C')
        
        # 绘制地表温度演化
        ax4.plot(time/365.0, temperature[:, 0], linewidth=2,
                label=f'B = {B_val} °C')
        
        results['B_sensitivity'][B_val] = {
            'final_profile': temperature[-1, :],
            'surface_evolution': temperature[:, 0]
        }
    
    B = original_B  # 恢复原值
    
    ax3.set_xlabel('Temperature (°C)', fontsize=12)
    ax3.set_ylabel('Depth (m)', fontsize=12)
    ax3.set_title('Final Temperature Profiles\n(Different Surface Amplitude)', fontsize=13)
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4.set_xlabel('Time (years)', fontsize=12)
    ax4.set_ylabel('Surface Temperature (°C)', fontsize=12)
    ax4.set_title('Surface Temperature Evolution\n(Different Surface Amplitude)', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('surface_amplitude_sensitivity.png', dpi=300, bbox_inches='tight')
    
    print("    敏感性分析图表已保存")
    
    return results


def create_animation(depth, time, temperature, save_path=None):
    """
    创建温度演化动画
    
    制作显示温度随时间变化的动画。
    
    参数:
        depth (ndarray): 深度数组
        time (ndarray): 时间数组
        temperature (ndarray): 温度矩阵
        save_path (str, optional): 保存动画的路径
    """
    try:
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 设置坐标轴范围
        temp_min = np.min(temperature) - 1
        temp_max = np.max(temperature) + 1
        
        line, = ax.plot([], [], 'b-', linewidth=2)
        ax.set_xlim(temp_min, temp_max)
        ax.set_ylim(depth[-1], depth[0])
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Depth (m)', fontsize=12)
        ax.set_title('Earth Crust Temperature Evolution', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加时间文本
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def animate(frame):
            # 每10天更新一次
            idx = frame * 10
            if idx >= len(time):
                idx = len(time) - 1
            
            line.set_data(temperature[idx, :], depth)
            time_text.set_text(f'Time: {time[idx]/365.0:.2f} years\nDay: {time[idx]:.0f}')
            return line, time_text
        
        # 创建动画
        frames = len(time) // 10
        anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            print(f"  动画已保存至: {save_path}")
        
        return anim
        
    except ImportError:
        print("  警告: 无法创建动画，缺少必要的库")
        return None


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
        print(f"   达到稳态时间: {evolution_results['steady_state_time']} 年")
        print(f"   能量守恒: {evolution_results['energy_conservation']['is_conserved']}")
        
        # 3. 绘制季节性温度轮廓
        print("\n3. 绘制季节性温度轮廓...")
        # 选择第10年的四个时间点
        year_10_start = int(9 * 365)  # 第10年开始
        season_times = [year_10_start + i * 91 for i in range(4)]  # 每3个月一个点
        season_profiles = temperature[season_times, :]
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        
        fig = plot_seasonal_profiles(depth, season_profiles, seasons, 
                                   'seasonal_temperature_profiles.png')
        
        # 4. 计算解析解并比较
        print("\n4. 计算解析解并比较...")
        surface_params = {'A': A, 'B': B, 'tau': TAU}
        T_analytical = analytical_solution(depth, time, surface_params)
        comparison_results = compare_with_analytical(depth, time, temperature, T_analytical)
        
        # 5. 参数敏感性分析
        print("\n5. 参数敏感性分析...")
        sensitivity_results = parameter_sensitivity_analysis()
        
        # 6. 创建动画（可选）
        print("\n6. 创建温度演化动画...")
        anim = create_animation(depth, time, temperature, 'temperature_evolution.gif')
        
        print("\n=" * 60)
        print("所有分析完成！生成的文件:")
        print("  - seasonal_temperature_profiles.png")
        print("  - numerical_analytical_comparison.png")
        print("  - thermal_diffusivity_sensitivity.png")
        print("  - surface_amplitude_sensitivity.png")
        if anim:
            print("  - temperature_evolution.gif")
        print("=" * 60)
        
        # 显示图表
        plt.show()
        
    except Exception as e:
        print(f"\n运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()