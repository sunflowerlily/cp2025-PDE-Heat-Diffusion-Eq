import unittest
import numpy as np
import os
import sys

# 添加父目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入参考答案
#from solution.heat_diffusion_solution import (
from heat_diffusion_student import (
    basic_heat_diffusion,
    analytical_solution,
    stability_analysis,
    different_initial_condition,
    heat_diffusion_with_cooling
)

class TestHeatDiffusion(unittest.TestCase):
    def setUp(self):
        """设置测试所需的公共参数"""
        self.dx = 0.01
        self.dt = 0.5
        self.Nx = 101  # L=1, dx=0.01
        self.Nt = 2000
        
    def test_basic_heat_diffusion_shape(self):
        """测试基本热传导模拟的输出形状"""
        u = basic_heat_diffusion()
        self.assertEqual(u.shape, (self.Nx, self.Nt))
        
    def test_basic_heat_diffusion_boundary(self):
        """测试边界条件是否正确应用"""
        u = basic_heat_diffusion()
        np.testing.assert_array_equal(u[0, :], 0)  # 左边界
        np.testing.assert_array_equal(u[-1, :], 0) # 右边界
        
    def test_analytical_solution_shape(self):
        """测试解析解的输出形状"""
        s = analytical_solution()
        self.assertEqual(s.shape, (self.Nx, self.Nt))
        
    def test_stability_analysis_unstable(self):
        """测试不稳定条件下的数值解"""
        # 这里我们主要检查函数是否能正常运行
        # 实际测试中应该检查数值是否发散
        self.assertIsNone(stability_analysis())
        
    def test_different_initial_condition(self):
        """测试不同初始条件的应用"""
        u = different_initial_condition()
        self.assertIsNotNone(u, "函数应返回计算结果")
        # 检查初始条件是否正确应用
        np.testing.assert_allclose(u[1:50, 0], 100, atol=1e-6)  # 左半部分(排除边界点)
        np.testing.assert_allclose(u[50:-1, 0], 50, atol=1e-6)  # 右半部分(排除边界点)
        self.assertEqual(u.shape, (self.Nx, 1000))
        
    def test_cooling_effect(self):
        """测试冷却效应是否应用"""
        # 主要检查函数是否能正常运行
        # 实际测试中应该检查温度是否比没有冷却时更低
        self.assertIsNone(heat_diffusion_with_cooling())

if __name__ == "__main__":
    unittest.main()