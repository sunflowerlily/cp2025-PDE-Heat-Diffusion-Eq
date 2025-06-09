#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学生模板：量子隧穿效应数值模拟
文件：quantum_tunneling_student.py
重要：函数名称必须与参考答案一致！

本项目使用Crank-Nicolson方法求解一维含时薛定谔方程，
模拟电子波包穿越方势垒的量子隧穿现象。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

# 物理常数（原子单位制）
HBAR = 1.0  # 约化普朗克常数
M_E = 1.0   # 电子质量
EV_TO_AU = 1.0/27.211386245988  # eV到原子单位的转换

class QuantumTunnelingSolver:
    """
    量子隧穿效应求解器
    
    使用Crank-Nicolson方法求解一维含时薛定谔方程：
    i*hbar * ∂ψ/∂t = -hbar²/(2m) * ∂²ψ/∂x² + V(x)*ψ
    
    属性:
        L (float): 计算区域长度
        nx (int): 空间网格点数
        nt (int): 时间步数
        dx (float): 空间步长
        dt (float): 时间步长
        x (ndarray): 空间坐标数组
        t (ndarray): 时间坐标数组
        V (ndarray): 势能函数
        psi (ndarray): 波函数数组 (nx, nt)
    """
    
    def __init__(self, L=20.0, nx=1000, t_final=50.0, nt=2000, 
                 barrier_height=0.18*EV_TO_AU, barrier_width=2.0, barrier_center=10.0):
        """
        初始化量子隧穿求解器
        
        参数:
            L (float): 计算区域长度 (原子单位)
            nx (int): 空间网格点数
            t_final (float): 总计算时间 (原子单位)
            nt (int): 时间步数
            barrier_height (float): 势垒高度 (原子单位)
            barrier_width (float): 势垒宽度 (原子单位)
            barrier_center (float): 势垒中心位置 (原子单位)
        
        物理背景:
        - 原子单位制：长度单位为Bohr半径，能量单位为Hartree
        - 势垒高度0.18 eV对应约0.0066 Hartree
        - 典型计算区域为几十个Bohr半径
        
        数值方法:
        - Crank-Nicolson隐式格式保证数值稳定性
        - 复数三对角矩阵系统的高效求解
        """
        self.L = L
        self.nx = nx
        self.nt = nt
        self.t_final = t_final
        
        # TODO: 初始化空间和时间网格
        # self.dx = 
        # self.dt = 
        # self.x = 
        # self.t = 
        
        # TODO: 设置势垒参数
        # self.barrier_height = 
        # self.barrier_width = 
        # self.barrier_center = 
        
        # TODO: 初始化势能函数和波函数数组
        # self.V = 
        # self.psi = 
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def setup_potential(self):
        """
        设置方势垒势能函数
        
        返回:
            ndarray: 势能函数 V(x)
        
        势垒模型:
        V(x) = 0           for x < x1 or x > x2
        V(x) = V0          for x1 ≤ x ≤ x2
        
        其中 x1, x2 是势垒的左右边界
        
        实现步骤:
        1. 计算势垒的左右边界位置
        2. 创建势能数组，初始化为零
        3. 在势垒区域设置势垒高度
        4. 返回势能函数
        """
        # TODO: 计算势垒边界
        # x1 = self.barrier_center - self.barrier_width / 2
        # x2 = self.barrier_center + self.barrier_width / 2
        
        # TODO: 创建势能数组
        # V = np.zeros(self.nx)
        
        # TODO: 设置势垒区域的势能
        # barrier_mask = (self.x >= x1) & (self.x <= x2)
        # V[barrier_mask] = self.barrier_height
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def setup_initial_wavepacket(self, x0=5.0, sigma=1.0, k0=2.0):
        """
        设置初始高斯波包
        
        参数:
            x0 (float): 波包中心位置
            sigma (float): 波包宽度参数
            k0 (float): 初始动量对应的波数
        
        返回:
            ndarray: 归一化的初始波函数
        
        高斯波包公式:
        ψ(x,0) = exp[-(x-x0)²/(2σ²)] * exp(ik0*x)
        
        物理意义:
        - 高斯包络描述粒子的位置不确定性
        - 平面波因子exp(ik0*x)描述初始动量
        - k0 = p0/ħ，其中p0是初始动量
        
        实现步骤:
        1. 计算高斯包络函数
        2. 计算平面波因子
        3. 组合得到复数波函数
        4. 归一化波函数
        5. 返回初始波函数
        """
        # TODO: 计算高斯包络
        # gaussian_envelope = np.exp(-(self.x - x0)**2 / (2 * sigma**2))
        
        # TODO: 计算平面波因子
        # plane_wave = np.exp(1j * k0 * self.x)
        
        # TODO: 组合波函数
        # psi_initial = gaussian_envelope * plane_wave
        
        # TODO: 归一化
        # norm = np.sqrt(np.trapz(np.abs(psi_initial)**2, self.x))
        # psi_initial = psi_initial / norm
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def build_crank_nicolson_matrices(self):
        """
        构建Crank-Nicolson格式的系数矩阵
        
        返回:
            tuple: (A_matrix, B_matrix) 左侧和右侧系数矩阵
        
        Crank-Nicolson格式:
        (I + iΔt/(2ħ) * H) * ψ^(n+1) = (I - iΔt/(2ħ) * H) * ψ^n
        
        其中 H 是哈密顿算符的离散化矩阵：
        H = -ħ²/(2m) * ∇² + V(x)
        
        离散化的动能算符（二阶中心差分）：
        ∇²ψ_j ≈ (ψ_{j+1} - 2ψ_j + ψ_{j-1}) / Δx²
        
        实现步骤:
        1. 计算动能系数 r = ħ²/(2m*Δx²)
        2. 构建动能算符的三对角矩阵
        3. 添加势能对角项
        4. 构建A和B矩阵
        5. 处理边界条件
        """
        # TODO: 计算动能系数
        # r = HBAR**2 / (2 * M_E * self.dx**2)
        
        # TODO: 构建动能算符的三对角矩阵
        # 主对角线: 2r
        # 上下对角线: -r
        # kinetic_main = 2 * r * np.ones(self.nx)
        # kinetic_off = -r * np.ones(self.nx - 1)
        
        # TODO: 添加势能项
        # hamiltonian_main = kinetic_main + self.V
        
        # TODO: 构建A和B矩阵
        # 时间步长因子
        # alpha = 1j * self.dt / (2 * HBAR)
        
        # A矩阵 = I + α*H
        # B矩阵 = I - α*H
        
        # TODO: 使用scipy.sparse.diags构建稀疏矩阵
        
        # TODO: 处理边界条件（吸收边界或固定边界）
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def solve_time_evolution(self):
        """
        求解波函数的时间演化
        
        返回:
            ndarray: 波函数演化数组 psi(x,t)
        
        算法流程:
        1. 设置初始波函数
        2. 构建Crank-Nicolson矩阵
        3. 时间循环求解线性方程组
        4. 存储每个时间步的波函数
        
        数值稳定性:
        - Crank-Nicolson格式是无条件稳定的
        - 保持波函数的归一化（在数值误差范围内）
        - 保持概率守恒
        
        实现步骤:
        1. 初始化波函数数组
        2. 设置初始条件
        3. 构建系数矩阵
        4. 时间步进循环
        5. 求解线性方程组
        6. 更新波函数
        """
        # TODO: 设置初始波函数
        # psi_initial = self.setup_initial_wavepacket()
        # self.psi[:, 0] = psi_initial
        
        # TODO: 构建Crank-Nicolson矩阵
        # A_matrix, B_matrix = self.build_crank_nicolson_matrices()
        
        # TODO: 时间演化循环
        # for n in range(self.nt - 1):
        #     # 计算右侧向量
        #     rhs = B_matrix @ self.psi[:, n]
        #     
        #     # 求解线性方程组
        #     self.psi[:, n+1] = spsolve(A_matrix, rhs)
        #     
        #     # 可选：检查归一化
        #     if n % 100 == 0:
        #         norm = np.trapz(np.abs(self.psi[:, n+1])**2, self.x)
        #         print(f"Time step {n}: Norm = {norm:.6f}")
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def calculate_probability_density(self):
        """
        计算概率密度 |ψ(x,t)|²
        
        返回:
            ndarray: 概率密度数组
        
        物理意义:
        - |ψ(x,t)|² 表示在位置x、时刻t找到粒子的概率密度
        - 积分 ∫|ψ(x,t)|²dx = 1 (归一化条件)
        
        实现步骤:
        1. 计算波函数的模长平方
        2. 返回概率密度数组
        """
        # TODO: 计算概率密度
        # probability_density = np.abs(self.psi)**2
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def calculate_probability_current(self):
        """
        计算概率流密度 j(x,t)
        
        返回:
            ndarray: 概率流密度数组
        
        概率流密度公式:
        j(x,t) = (ħ/2mi) * [ψ* ∂ψ/∂x - ψ ∂ψ*/∂x]
        
        物理意义:
        - 描述概率在空间中的流动
        - 满足连续性方程：∂|ψ|²/∂t + ∂j/∂x = 0
        - 正值表示向右流动，负值表示向左流动
        
        数值实现:
        - 使用中心差分计算空间导数
        - 处理边界点的特殊情况
        
        实现步骤:
        1. 计算波函数的空间导数
        2. 计算概率流密度
        3. 处理边界条件
        4. 返回流密度数组
        """
        # TODO: 初始化流密度数组
        # current = np.zeros_like(self.psi, dtype=float)
        
        # TODO: 计算空间导数（中心差分）
        # for n in range(self.nt):
        #     psi_n = self.psi[:, n]
        #     
        #     # 内部点使用中心差分
        #     dpsi_dx = np.zeros_like(psi_n)
        #     dpsi_dx[1:-1] = (psi_n[2:] - psi_n[:-2]) / (2 * self.dx)
        #     
        #     # 边界点使用前向/后向差分
        #     dpsi_dx[0] = (psi_n[1] - psi_n[0]) / self.dx
        #     dpsi_dx[-1] = (psi_n[-1] - psi_n[-2]) / self.dx
        #     
        #     # 计算概率流密度
        #     current[:, n] = (HBAR / (2j * M_E)) * (np.conj(psi_n) * dpsi_dx - psi_n * np.conj(dpsi_dx))
        #     current[:, n] = np.real(current[:, n])  # 取实部
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def analyze_tunneling_effect(self):
        """
        分析隧穿效应
        
        返回:
            dict: 包含透射系数、反射系数等分析结果
        
        分析内容:
        1. 透射系数：穿过势垒的概率
        2. 反射系数：被势垒反射的概率
        3. 隧穿时间：波包穿越势垒的特征时间
        4. 能量分析：与经典预期的比较
        
        计算方法:
        - 将计算区域分为三部分：入射区、势垒区、透射区
        - 在足够长时间后计算各区域的概率
        - 分析波包的形状变化
        
        实现步骤:
        1. 定义区域边界
        2. 计算各区域的概率
        3. 分析时间演化特征
        4. 计算物理量
        5. 返回分析结果
        """
        # TODO: 定义区域边界
        # barrier_left = self.barrier_center - self.barrier_width / 2
        # barrier_right = self.barrier_center + self.barrier_width / 2
        
        # TODO: 找到对应的网格索引
        # idx_left = np.argmin(np.abs(self.x - barrier_left))
        # idx_right = np.argmin(np.abs(self.x - barrier_right))
        
        # TODO: 计算最终时刻各区域的概率
        # prob_density_final = self.calculate_probability_density()[:, -1]
        
        # incident_prob = np.trapz(prob_density_final[:idx_left], self.x[:idx_left])
        # barrier_prob = np.trapz(prob_density_final[idx_left:idx_right], self.x[idx_left:idx_right])
        # transmitted_prob = np.trapz(prob_density_final[idx_right:], self.x[idx_right:])
        
        # TODO: 计算透射和反射系数
        # transmission_coeff = transmitted_prob
        # reflection_coeff = incident_prob
        
        # TODO: 分析结果
        # results = {
        #     'transmission_coefficient': transmission_coeff,
        #     'reflection_coefficient': reflection_coeff,
        #     'barrier_probability': barrier_prob,
        #     'total_probability': incident_prob + barrier_prob + transmitted_prob
        # }
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def plot_wavefunction_evolution(self, time_indices=None, save_fig=False):
        """
        绘制波函数演化图
        
        参数:
            time_indices (list): 要绘制的时间索引列表
            save_fig (bool): 是否保存图片
        
        绘制内容:
        1. 概率密度 |ψ(x,t)|² 的时间演化
        2. 势能函数的叠加显示
        3. 多个时刻的对比
        4. 清晰的图例和标注
        
        实现步骤:
        1. 设置默认时间索引
        2. 计算概率密度
        3. 创建图形和坐标轴
        4. 绘制势能函数
        5. 绘制不同时刻的概率密度
        6. 添加图例和标注
        7. 保存图片（可选）
        """
        if time_indices is None:
            time_indices = [0, self.nt//4, self.nt//2, 3*self.nt//4, self.nt-1]
        
        # TODO: 计算概率密度
        # prob_density = self.calculate_probability_density()
        
        # TODO: 创建图形
        # plt.figure(figsize=(12, 8))
        
        # TODO: 绘制势能函数
        # plt.subplot(2, 1, 1)
        # plt.plot(self.x, self.V, 'k-', linewidth=2, label='Potential V(x)')
        # plt.ylabel('Energy (a.u.)')
        # plt.title('Quantum Tunneling: Wavefunction Evolution')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        
        # TODO: 绘制概率密度演化
        # plt.subplot(2, 1, 2)
        # colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
        # 
        # for i, t_idx in enumerate(time_indices):
        #     time_value = self.t[t_idx]
        #     plt.plot(self.x, prob_density[:, t_idx], 
        #              color=colors[i], linewidth=2, 
        #              label=f't = {time_value:.2f} a.u.')
        # 
        # plt.xlabel('Position x (a.u.)')
        # plt.ylabel('Probability Density |ψ|²')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # 
        # plt.tight_layout()
        # 
        # if save_fig:
        #     plt.savefig('wavefunction_evolution.png', dpi=300, bbox_inches='tight')
        # 
        # plt.show()
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def create_animation(self, filename='tunneling_animation.gif', interval=50):
        """
        创建波包演化动画
        
        参数:
            filename (str): 动画文件名
            interval (int): 帧间隔（毫秒）
        
        动画内容:
        1. 实时显示概率密度的变化
        2. 势垒的固定显示
        3. 时间标签的更新
        4. 流畅的动画效果
        
        实现步骤:
        1. 计算概率密度
        2. 设置图形和坐标轴
        3. 定义动画更新函数
        4. 创建动画对象
        5. 保存动画文件
        """
        # TODO: 计算概率密度
        # prob_density = self.calculate_probability_density()
        
        # TODO: 设置图形
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # TODO: 绘制势能函数（静态）
        # ax1.plot(self.x, self.V, 'k-', linewidth=2)
        # ax1.set_ylabel('Potential V(x)')
        # ax1.set_title('Quantum Tunneling Animation')
        # ax1.grid(True, alpha=0.3)
        
        # TODO: 初始化概率密度图线
        # line, = ax2.plot([], [], 'b-', linewidth=2)
        # ax2.set_xlim(self.x[0], self.x[-1])
        # ax2.set_ylim(0, np.max(prob_density) * 1.1)
        # ax2.set_xlabel('Position x (a.u.)')
        # ax2.set_ylabel('Probability Density |ψ|²')
        # ax2.grid(True, alpha=0.3)
        
        # TODO: 添加时间文本
        # time_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
        
        # TODO: 定义动画更新函数
        # def animate(frame):
        #     line.set_data(self.x, prob_density[:, frame])
        #     time_text.set_text(f'Time = {self.t[frame]:.2f} a.u.')
        #     return line, time_text
        
        # TODO: 创建动画
        # ani = animation.FuncAnimation(fig, animate, frames=self.nt,
        #                              interval=interval, blit=True, repeat=True)
        
        # TODO: 保存动画
        # ani.save(filename, writer='pillow', fps=1000//interval)
        # print(f"Animation saved as {filename}")
        
        # plt.show()
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def run_full_simulation(self):
        """
        运行完整的量子隧穿模拟
        
        返回:
            dict: 包含所有分析结果的字典
        
        模拟流程:
        1. 初始化系统
        2. 求解时间演化
        3. 分析隧穿效应
        4. 生成可视化结果
        5. 返回分析报告
        
        实现步骤:
        1. 设置势能函数
        2. 求解波函数演化
        3. 计算物理量
        4. 进行隧穿分析
        5. 生成图表
        6. 返回结果
        """
        print("开始量子隧穿模拟...")
        
        # TODO: 设置势能函数
        # self.V = self.setup_potential()
        # print("势能函数设置完成")
        
        # TODO: 求解时间演化
        # self.solve_time_evolution()
        # print("时间演化求解完成")
        
        # TODO: 分析隧穿效应
        # tunneling_results = self.analyze_tunneling_effect()
        # print("隧穿效应分析完成")
        
        # TODO: 生成可视化
        # self.plot_wavefunction_evolution(save_fig=True)
        # print("波函数演化图生成完成")
        
        # TODO: 创建动画（可选）
        # self.create_animation()
        # print("动画创建完成")
        
        # TODO: 返回结果
        # results = {
        #     'solver_parameters': {
        #         'L': self.L,
        #         'nx': self.nx,
        #         'nt': self.nt,
        #         'dx': self.dx,
        #         'dt': self.dt
        #     },
        #     'physical_parameters': {
        #         'barrier_height': self.barrier_height,
        #         'barrier_width': self.barrier_width,
        #         'barrier_center': self.barrier_center
        #     },
        #     'tunneling_analysis': tunneling_results
        # }
        
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def compare_different_energies(energies, barrier_height=0.18*EV_TO_AU):
    """
    比较不同初始能量下的隧穿效应
    
    参数:
        energies (list): 不同的初始动能列表
        barrier_height (float): 势垒高度
    
    返回:
        dict: 不同能量下的隧穿系数
    
    物理背景:
    - 经典粒子：E < V0 时无法穿越势垒
    - 量子粒子：即使 E < V0 也有隧穿概率
    - 隧穿概率随能量增加而增大
    
    实现步骤:
    1. 遍历不同的初始能量
    2. 计算对应的初始动量
    3. 运行隧穿模拟
    4. 记录透射系数
    5. 分析能量依赖性
    """
    # TODO: 初始化结果字典
    # results = {'energies': [], 'transmission_coefficients': []}
    
    # TODO: 遍历不同能量
    # for E in energies:
    #     # 计算对应的波数 k = sqrt(2mE)/ħ
    #     k0 = np.sqrt(2 * M_E * E) / HBAR
    #     
    #     # 创建求解器
    #     solver = QuantumTunnelingSolver(barrier_height=barrier_height)
    #     
    #     # 运行模拟
    #     simulation_results = solver.run_full_simulation()
    #     
    #     # 记录结果
    #     transmission_coeff = simulation_results['tunneling_analysis']['transmission_coefficient']
    #     results['energies'].append(E)
    #     results['transmission_coefficients'].append(transmission_coeff)
    #     
    #     print(f"Energy: {E:.4f} a.u., Transmission: {transmission_coeff:.4f}")
    
    # TODO: 绘制能量依赖性图
    # plt.figure(figsize=(10, 6))
    # plt.plot(results['energies'], results['transmission_coefficients'], 
    #          'bo-', linewidth=2, markersize=8)
    # plt.axvline(x=barrier_height, color='r', linestyle='--', 
    #            label=f'Barrier Height = {barrier_height:.4f} a.u.')
    # plt.xlabel('Initial Energy (a.u.)')
    # plt.ylabel('Transmission Coefficient')
    # plt.title('Quantum Tunneling: Energy Dependence')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.savefig('energy_dependence.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def verify_conservation_laws(solver):
    """
    验证守恒定律
    
    参数:
        solver (QuantumTunnelingSolver): 已完成计算的求解器
    
    返回:
        dict: 守恒定律验证结果
    
    验证内容:
    1. 概率守恒：∫|ψ|²dx = 1
    2. 连续性方程：∂|ψ|²/∂t + ∂j/∂x = 0
    3. 能量守恒（在无耗散系统中）
    
    实现步骤:
    1. 计算每个时刻的总概率
    2. 验证连续性方程
    3. 检查能量守恒
    4. 生成验证报告
    """
    # TODO: 计算概率密度和概率流
    # prob_density = solver.calculate_probability_density()
    # prob_current = solver.calculate_probability_current()
    
    # TODO: 验证概率守恒
    # total_prob = np.zeros(solver.nt)
    # for n in range(solver.nt):
    #     total_prob[n] = np.trapz(prob_density[:, n], solver.x)
    
    # TODO: 验证连续性方程
    # continuity_error = np.zeros((solver.nx, solver.nt-1))
    # for n in range(solver.nt-1):
    #     # 时间导数
    #     dpdt = (prob_density[:, n+1] - prob_density[:, n]) / solver.dt
    #     
    #     # 空间导数
    #     djdx = np.zeros(solver.nx)
    #     djdx[1:-1] = (prob_current[2:, n] - prob_current[:-2, n]) / (2 * solver.dx)
    #     
    #     # 连续性方程误差
    #     continuity_error[:, n] = dpdt + djdx
    
    # TODO: 生成验证结果
    # results = {
    #     'probability_conservation': {
    #         'mean_total_probability': np.mean(total_prob),
    #         'std_total_probability': np.std(total_prob),
    #         'max_deviation': np.max(np.abs(total_prob - 1.0))
    #     },
    #     'continuity_equation': {
    #         'max_error': np.max(np.abs(continuity_error)),
    #         'rms_error': np.sqrt(np.mean(continuity_error**2))
    #     }
    # }
    
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

if __name__ == "__main__":
    # 示例使用
    print("量子隧穿效应数值模拟")
    print("=" * 50)
    
    # TODO: 创建求解器实例
    # solver = QuantumTunnelingSolver(
    #     L=20.0,           # 计算区域长度
    #     nx=1000,          # 空间网格点数
    #     t_final=50.0,     # 总时间
    #     nt=2000,          # 时间步数
    #     barrier_height=0.18*EV_TO_AU,  # 势垒高度
    #     barrier_width=2.0,             # 势垒宽度
    #     barrier_center=10.0             # 势垒位置
    # )
    
    # TODO: 运行完整模拟
    # results = solver.run_full_simulation()
    
    # TODO: 打印结果
    # print("\n模拟结果:")
    # print(f"透射系数: {results['tunneling_analysis']['transmission_coefficient']:.4f}")
    # print(f"反射系数: {results['tunneling_analysis']['reflection_coefficient']:.4f}")
    
    # TODO: 验证守恒定律
    # conservation_results = verify_conservation_laws(solver)
    # print("\n守恒定律验证:")
    # print(f"概率守恒偏差: {conservation_results['probability_conservation']['max_deviation']:.6f}")
    
    # TODO: 比较不同能量
    # energies = np.linspace(0.05, 0.25, 10) * EV_TO_AU
    # energy_results = compare_different_energies(energies)
    
    print("\n请实现上述函数以完成量子隧穿模拟！")
    print("参考课件中的三对角矩阵算法和教材P338页的程序。")