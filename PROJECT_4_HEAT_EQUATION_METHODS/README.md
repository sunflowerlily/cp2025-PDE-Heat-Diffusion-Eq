# 项目4：有限长细杆热传导方程多种数值方法比较

**课程：** 计算物理 **项目类型：** 数值方法比较研究 **难度：** ⭐⭐⭐⭐

## 项目概述

本项目要求学生实现和比较四种不同的数值方法来求解一维热传导方程，深入理解各种数值格式的特点、稳定性条件和精度特性。这是一个综合性项目，涵盖了偏微分方程数值解的核心概念。

## 学习目标

- 掌握显式和隐式差分格式的实现
- 理解数值稳定性的概念和重要性
- 学会使用稀疏矩阵求解线性方程组
- 熟悉高级ODE求解器的使用
- 培养数值方法比较和分析能力

## 问题描述

### 控制方程
一维热传导方程：
$$\frac{\partial u}{\partial t} = \alpha^2 \frac{\partial^2 u}{\partial x^2}$$

### 边界和初始条件
- **定义域**：$x \in [0, 20]$，$t \in [0, 25]$
- **边界条件**：$u(0,t) = u(20,t) = 0$（两端温度固定为0）
- **初始条件**：$u(x,0) = \begin{cases} 1 & \text{if } 10 \leq x \leq 11 \\ 0 & \text{elsewhere} \end{cases}$
- **物理参数**：$\alpha^2 = 10$（热扩散系数）

## 实现方法

### 1. 显式差分法 (FTCS)
- **格式**：前向时间，中心空间差分
- **稳定性**：条件稳定，$r = \frac{\alpha^2 \Delta t}{(\Delta x)^2} \leq 0.5$
- **精度**：时间一阶，空间二阶
- **特点**：实现简单，计算快速，但有稳定性限制

### 2. 隐式差分法 (后向欧拉)
- **格式**：后向时间，中心空间差分
- **稳定性**：无条件稳定
- **精度**：时间一阶，空间二阶
- **特点**：需要求解线性方程组，稳定性好

### 3. Crank-Nicolson方法
- **格式**：隐式和显式的平均
- **稳定性**：无条件稳定
- **精度**：时间二阶，空间二阶
- **特点**：最高精度，被广泛使用

### 4. scipy.solve_ivp方法
- **格式**：将PDE转换为ODE系统
- **稳定性**：取决于选择的积分器
- **精度**：可变，支持自适应步长
- **特点**：使用高级求解器，精度可控

## 文件结构

```
PROJECT_4_HEAT_EQUATION_METHODS/
├── README.md                           # 项目说明（本文件）
├── 项目说明.md                         # 详细项目要求
├── 实验报告模板.md                     # 报告模板
├── solution/
│   └── heat_equation_methods_solution.py  # 参考答案
├── tests/
│   └── test_heat_equation_methods.py      # 测试文件
└── heat_equation_methods_student.py       # 学生模板
```

## 快速开始

### 1. 环境要求
```bash
pip install numpy scipy matplotlib
```

### 2. 运行学生模板
```bash
python heat_equation_methods_student.py
```

### 3. 运行测试
```bash
python tests/test_heat_equation_methods.py
```

### 4. 查看参考答案（仅供验证）
```bash
python solution/heat_equation_methods_solution.py
```

## 实现指南

### 核心函数列表

1. **`create_initial_condition(x)`** - 创建初始条件
2. **`solve_ftcs(nx, nt, total_time)`** - FTCS显式方法
3. **`solve_backward_euler(nx, nt, total_time)`** - 后向欧拉隐式方法
4. **`solve_crank_nicolson(nx, nt, total_time)`** - Crank-Nicolson方法
5. **`solve_with_scipy(nx, total_time)`** - scipy.solve_ivp方法
6. **`calculate_errors(u_numerical, u_reference, dx)`** - 误差计算
7. **`compare_methods(nx, nt)`** - 方法比较
8. **`plot_comparison(results)`** - 结果可视化

### 实现顺序建议

1. **第一步**：实现初始条件函数
2. **第二步**：实现FTCS显式方法（最简单）
3. **第三步**：实现后向欧拉方法（学习稀疏矩阵）
4. **第四步**：实现Crank-Nicolson方法
5. **第五步**：实现scipy方法
6. **第六步**：实现误差计算和比较函数
7. **第七步**：完善可视化功能

### 关键技术点

#### 稀疏矩阵构建
```python
from scipy.sparse import diags
# 构建三对角矩阵
diagonals = [上对角线, 主对角线, 下对角线]
offsets = [1, 0, -1]
A = diags(diagonals, offsets, format='csr')
```

#### 线性方程组求解
```python
from scipy.sparse.linalg import spsolve
u_new = spsolve(A, rhs)
```

#### ODE系统定义
```python
def ode_system(t, u_interior):
    return alpha * A_spatial @ u_interior
```

## 评分标准

| 功能模块 | 分值 | 评分要点 |
|----------|------|----------|
| 初始条件 | 5分 | 正确实现分段函数 |
| FTCS方法 | 8分 | 算法正确，稳定性检查 |
| 后向欧拉 | 8分 | 矩阵构建，方程组求解 |
| Crank-Nicolson | 8分 | 高精度格式实现 |
| scipy方法 | 6分 | ODE转换，求解器使用 |
| 误差计算 | 3分 | 多种误差范数 |
| 边界条件 | 4分 | 所有方法正确处理 |
| 稳定性分析 | 3分 | FTCS稳定性条件 |
| 守恒性质 | 3分 | 物理量守恒检验 |
| 方法比较 | 2分 | 综合比较功能 |
| 一致性检验 | 5分 | 不同方法结果一致 |
| **总计** | **55分** | |
| 实验报告 | 15分 | 分析深度，结果讨论 |
| **项目总分** | **70分** | |

### 加分项（最多5分）
- 实现多种scipy积分器比较 (+2分)
- 收敛性研究和可视化 (+2分)
- 性能优化和并行化 (+1分)

## 常见问题

### Q1: FTCS方法出现数值不稳定怎么办？
**A**: 检查稳定性条件 $r \leq 0.5$，减小时间步长或增加空间网格点数。

### Q2: 稀疏矩阵构建困难？
**A**: 使用`scipy.sparse.diags`，注意对角线元素的符号和位置。

### Q3: scipy.solve_ivp结果与其他方法差异很大？
**A**: 检查ODE系统定义，确保空间离散化正确，调整容差参数。

### Q4: 边界条件如何处理？
**A**: 在每个时间步后强制设置边界点为0，或在矩阵构建时直接处理。

### Q5: 如何验证实现正确性？
**A**: 运行测试文件，检查能量守恒，比较不同方法的结果一致性。

## 扩展思考

1. **二维扩展**：如何将这些方法扩展到二维热传导方程？
2. **非线性问题**：对于非线性热传导，需要如何修改算法？
3. **自适应网格**：如何实现自适应网格细化？
4. **并行计算**：哪些部分可以并行化以提高效率？
5. **实际应用**：在工程中如何选择合适的数值方法？

## 参考资源

- **教材**：《计算物理学》相关章节
- **文档**：[SciPy稀疏矩阵文档](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- **文档**：[SciPy ODE求解器文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
- **论文**：Crank-Nicolson方法的经典论文

---

**项目完成时间建议：** 2-3周  
**难度评估：** 需要扎实的数值分析基础和Python编程能力  
**学习收获：** 深入理解PDE数值解的核心概念和实践技能