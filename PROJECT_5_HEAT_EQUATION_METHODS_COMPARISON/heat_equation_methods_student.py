#!/usr/bin/env python3
"""
学生模板：热传导方程多种数值方法比较
文件：heat_equation_methods_student.py
重要：函数名称必须与参考答案一致！

实现6种不同的数值方法求解一维热传导方程：
1. FTCS (Forward Time Central Space) - 显式格式
2. Laplace算符显式方法
3. BTCS (Backward Time Central Space) - 隐式格式
4. Crank-Nicolson方法
5. 变形Crank-Nicolson方法
6. solve_ivp方法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time
from typing import Callable, Tuple, Dict, List

class heat_equation_solver:
    """
    一维热传导方程求解器，实现多种数值方法
    
    求解方程: u_t = a^2 * u_xx，边界条件: u(0,t) = u(L,t) = 0
    """
    
    def __init__(self, L: float = 20.0, T: float = 25.0, a_squared: float = 10.0, 
                 nx: int = 100, nt: int = 1000):
        """
        初始化热传导方程求解器
        
        参数:
            L: 杆的长度
            T: 总时间
            a_squared: 热扩散系数
            nx: 空间网格点数
            nt: 时间步数
            
        物理背景: 热传导方程描述热量在物体中的传播过程
        数值方法: 使用有限差分方法离散化偏微分方程
        
        实现步骤:
        1. 设置网格参数
        2. 计算空间步长dx和时间步长dt
        3. 计算稳定性参数r = a^2 * dt / dx^2
        4. 初始化网格坐标
        """
        # TODO: 设置基本参数
        # TODO: 计算网格步长
        # TODO: 创建空间和时间坐标数组
        # TODO: 计算稳定性参数
        # TODO: 初始化初始条件相关变量
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def set_initial_condition(self, phi_func: Callable[[np.ndarray], np.ndarray]):
        """
        设置初始条件函数
        
        参数:
            phi_func: 接受x数组并返回初始温度分布的函数
            
        物理背景: 初始条件描述t=0时刻的温度分布
        
        实现步骤:
        1. 保存初始条件函数
        2. 计算初始温度分布
        3. 应用边界条件
        """
        # TODO: 保存初始条件函数
        # TODO: 计算初始温度分布
        # TODO: 应用边界条件 u(0) = u(L) = 0
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def ftcs_method(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward Time Central Space (FTCS) 显式方法
        
        返回:
            Tuple[时间数组, 解矩阵]
            
        物理背景: 显式格式在时间方向向前差分，空间方向中心差分
        数值方法: u_i^{n+1} = u_i^n + r*(u_{i+1}^n - 2*u_i^n + u_{i-1}^n)
        
        稳定性条件: r = a^2*dt/dx^2 <= 0.5
        
        实现步骤:
        1. 检查初始条件是否设置
        2. 初始化解矩阵
        3. 时间循环：对每个内部点应用FTCS格式
        4. 应用边界条件
        """
        # TODO: 检查初始条件
        # TODO: 初始化解矩阵
        # TODO: 实现时间循环
        # TODO: 在每个时间步应用FTCS格式
        # TODO: 应用边界条件
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def laplace_explicit_method(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用Laplace算符矩阵的显式方法
        
        返回:
            Tuple[时间数组, 解矩阵]
            
        物理背景: 使用矩阵形式表示Laplace算符，便于向量化计算
        数值方法: u^{n+1} = u^n + a^2*dt*L*u^n，其中L是Laplace算符矩阵
        
        实现步骤:
        1. 构建Laplace算符矩阵（三对角矩阵）
        2. 应用边界条件到矩阵
        3. 时间循环：矩阵向量乘法更新解
        """
        # TODO: 检查初始条件
        # TODO: 构建Laplace算符矩阵（使用diags函数）
        # TODO: 应用边界条件到矩阵
        # TODO: 时间循环：u^{n+1} = u^n + a^2*dt*L*u^n
        # TODO: 确保边界条件
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def btcs_method(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward Time Central Space (BTCS) 隐式方法
        
        返回:
            Tuple[时间数组, 解矩阵]
            
        物理背景: 隐式格式无条件稳定，但需要求解线性方程组
        数值方法: (I - r*L)*u^{n+1} = u^n
        
        实现步骤:
        1. 构建系数矩阵（三对角矩阵）
        2. 应用边界条件
        3. 时间循环：求解线性方程组
        """
        # TODO: 检查初始条件
        # TODO: 构建隐式格式的系数矩阵
        # TODO: 应用边界条件
        # TODO: 创建带状矩阵格式用于solve_banded
        # TODO: 时间循环：求解线性方程组
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def crank_nicolson_method(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crank-Nicolson方法 (theta = 0.5)
        
        返回:
            Tuple[时间数组, 解矩阵]
            
        物理背景: 半隐式格式，在时间方向使用中心差分，二阶精度
        数值方法: (I + r/2*L)*u^{n+1} = (I - r/2*L)*u^n
        
        优点: 二阶时间精度，无条件稳定
        
        实现步骤:
        1. 构建左端和右端系数矩阵
        2. 应用边界条件
        3. 时间循环：求解线性方程组
        """
        # TODO: 检查初始条件
        # TODO: 构建左端系数矩阵 (I + r/2*L)
        # TODO: 构建右端系数矩阵 (I - r/2*L)
        # TODO: 应用边界条件
        # TODO: 时间循环：求解 LHS*u^{n+1} = RHS*u^n
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def modified_crank_nicolson_method(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        变形Crank-Nicolson方法，使用theta = 0.6提高稳定性
        
        返回:
            Tuple[时间数组, 解矩阵]
            
        物理背景: 通过调整theta参数平衡精度和稳定性
        数值方法: (I + theta*r*L)*u^{n+1} = (I - (1-theta)*r*L)*u^n
        
        实现步骤:
        1. 设置theta = 0.6
        2. 构建修改的系数矩阵
        3. 求解线性方程组
        """
        # TODO: 设置theta参数
        # TODO: 构建修改的左端和右端矩阵
        # TODO: 应用边界条件
        # TODO: 时间循环求解
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def solve_ivp_method(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用scipy.integrate.solve_ivp求解热传导方程
        
        返回:
            Tuple[时间数组, 解矩阵]
            
        物理背景: 将PDE转换为ODE系统，利用高精度ODE求解器
        数值方法: du/dt = a^2 * d^2u/dx^2，转换为ODE系统
        
        实现步骤:
        1. 定义ODE系统函数
        2. 使用solve_ivp求解
        3. 处理返回结果
        """
        # TODO: 检查初始条件
        # TODO: 定义ODE系统函数 heat_ode(t, u)
        # TODO: 在ODE函数中实现空间二阶导数
        # TODO: 应用边界条件
        # TODO: 使用solve_ivp求解
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def analytical_solution(self, x: np.ndarray, t: float, n_terms: int = 100) -> np.ndarray:
        """
        使用Fourier级数的解析解（用于验证）
        
        参数:
            x: 空间坐标
            t: 时间
            n_terms: Fourier级数项数
            
        返回:
            给定x和t的解析解
            
        物理背景: 对于给定初始条件，可以用分离变量法得到Fourier级数解
        数学方法: u(x,t) = Σ b_n * sin(nπx/L) * exp(-a^2*(nπ/L)^2*t)
        
        实现步骤:
        1. 计算Fourier系数
        2. 求和Fourier级数
        3. 返回解析解
        """
        # TODO: 初始化解数组
        # TODO: 计算Fourier级数
        # TODO: 对于给定初始条件计算Fourier系数
        # TODO: 累加级数项
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def compare_methods(self) -> Dict[str, Dict]:
        """
        比较所有6种方法的精度和计算时间
        
        返回:
            包含每种方法结果和计时的字典
            
        实现步骤:
        1. 定义所有方法的字典
        2. 循环运行每种方法
        3. 计算误差和运行时间
        4. 返回比较结果
        """
        # TODO: 检查初始条件
        # TODO: 定义方法字典
        # TODO: 循环运行每种方法
        # TODO: 计算与解析解的误差
        # TODO: 记录计算时间
        # TODO: 处理异常情况
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def plot_comparison(self, results: Dict[str, Dict], save_fig: bool = True):
        """
        绘制所有方法的比较图
        
        参数:
            results: compare_methods()的结果
            save_fig: 是否保存图片
        """
        # TODO: 创建子图
        # TODO: 绘制解析解作为参考
        # TODO: 绘制每种方法的数值解
        # TODO: 添加误差和时间信息
        # TODO: 设置图例和标签
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def plot_error_analysis(self, results: Dict[str, Dict], save_fig: bool = True):
        """
        绘制所有方法的误差分析图
        
        参数:
            results: compare_methods()的结果
            save_fig: 是否保存图片
        """
        # TODO: 提取成功方法的数据
        # TODO: 创建误差比较柱状图
        # TODO: 创建计算时间比较图
        # TODO: 设置对数坐标和标签
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")


def initial_condition_function(x: np.ndarray) -> np.ndarray:
    """
    初始条件函数: phi(x) = 1 for 10 <= x <= 11, 0 otherwise
    
    参数:
        x: 空间坐标数组
        
    返回:
        初始温度分布
    """
    # TODO: 创建零数组
    # TODO: 设置10 <= x <= 11区域的值为1
    # TODO: 返回初始条件
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")


if __name__ == "__main__":
    # 问题参数
    L = 20.0      # 杆长
    T = 25.0      # 总时间
    a_squared = 10.0  # 热扩散系数
    nx = 100      # 空间网格点数
    nt = 1000     # 时间步数
    
    print("=" * 60)
    print("热传导方程求解器 - 多种方法比较")
    print("=" * 60)
    
    # 创建求解器
    solver = heat_equation_solver(L, T, a_squared, nx, nt)
    
    # 设置初始条件
    solver.set_initial_condition(initial_condition_function)
    
    print("\n初始条件设置: phi(x) = 1 for 10 <= x <= 11, 0 otherwise")
    print(f"稳定性参数 r = {solver.r:.4f}")
    
    # 比较所有方法
    print("\n运行所有6种方法的比较...")
    results = solver.compare_methods()
    
    # 打印总结
    print("\n" + "=" * 60)
    print("结果总结")
    print("=" * 60)
    
    # 生成比较图
    print("\n生成比较图...")
    solver.plot_comparison(results)
    solver.plot_error_analysis(results)
    
    print("\n分析完成！")