#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地壳热扩散数值模拟测试

简化版测试套件，仅包含核心功能测试
"""

import unittest
import numpy as np
import os
import sys

# 添加父目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from solution.earth_crust_diffusion_solution import solve_earth_crust_diffusion
from earth_crust_diffusion_student import solve_earth_crust_diffusion

# 物理常数
D = 0.1  # 热扩散率 (m^2/day)
A = 10.0  # 年平均地表温度 (°C)
B = 12.0  # 地表温度振幅 (°C)
TAU = 365.0  # 年周期 (days)
T_BOTTOM = 11.0  # 20米深处温度 (°C)
DEPTH_MAX = 20.0  # 最大深度 (m)

class TestEarthCrustDiffusion(unittest.TestCase):
    
    def setUp(self):
        self.depth, self.T = solve_earth_crust_diffusion()
    
    def test_solution_shape(self):
        """测试返回矩阵形状"""
        self.assertEqual(self.T.shape, (21, 366))
    
    def test_boundary_conditions(self):
        """测试边界条件"""
        # 测试地表边界条件
        self.assertTrue(all(self.T[0, :] >= A - B))
        self.assertTrue(all(self.T[0, :] <= A + B))
        
        # 测试底部边界条件
        self.assertTrue(all(self.T[-1, :] == T_BOTTOM))
    
    def test_temperature_range(self):
        """测试温度值物理合理性"""
        self.assertTrue(np.all(np.isfinite(self.T)))
        self.assertTrue(np.all(self.T >= -50))
        self.assertTrue(np.all(self.T <= 50))

if __name__ == '__main__':
    unittest.main()

