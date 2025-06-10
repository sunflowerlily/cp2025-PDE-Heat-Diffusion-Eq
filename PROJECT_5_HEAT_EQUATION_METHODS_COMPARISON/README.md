# 项目5：热传导方程多种数值方法比较

**课程：** 计算物理 **主题：** 热传导方程的数值求解方法比较

## 学习目标
- 掌握热传导方程的6种不同数值求解方法
- 理解显式、隐式和半隐式格式的稳定性特点
- 比较不同方法的精度和计算效率
- 学习使用scipy.integrate.solve_ivp求解偏微分方程

## 项目描述

求解有限长细杆的热传导问题：

$$
\begin{cases} 
u_t = a^2 u_{xx} \\
u(0, t) = 0, \quad u(l, t) = 0 \\
u(x, t = 0) = \varphi(x) 
\end{cases}
$$

其中：
- $l = 20$（杆长）
- $t = 25$（总时间）
- $a^2 = 10$（热扩散系数）
- 初始条件：$\varphi(x) = \begin{cases} 1 & (10 \leqslant x \leqslant 11) \\ 0 & (x < 10, x > 11) \end{cases}$

## 实现方法

构建`heat_equation_solver`类，包含以下6种数值方法：

1. **显式格式差分法 (FTCS)** - Forward Time Central Space
2. **Laplace算符显式格式** - 使用Laplace算符的显式方法
3. **隐式格式差分法 (BTCS)** - Backward Time Central Space
4. **Crank-Nicolson方法 (CN)** - 半隐式格式
5. **变形Crank-Nicolson方法** - 改进的CN方法
6. **solve_ivp方法** - 使用scipy的ODE求解器

## 技术要求

- Python 3.8+
- 依赖：numpy, scipy, matplotlib
- 实现模块化设计，每种方法独立实现
- 提供精度比较和可视化功能

## 评分标准

- 代码实现正确性：60%
- 方法比较分析：25%
- 代码质量和文档：15%

## 文件结构

```
PROJECT_5_HEAT_EQUATION_METHODS_COMPARISON/
├── README.md
├── heat_equation_methods_student.py
├── solution/
│   └── heat_equation_methods_solution.py
├── tests/
│   └── test_heat_equation_methods.py
├── 项目说明.md
└── 实验报告模板.md
```