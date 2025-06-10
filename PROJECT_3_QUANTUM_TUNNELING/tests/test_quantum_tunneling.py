import unittest
import numpy as np
import os
import sys

# 添加父目录到模块搜索路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from solution.quantum_tunneling_solution import QuantumTunnelingSolver
from quantum_tunneling_student import QuantumTunnelingSolver

class TestQuantumTunneling(unittest.TestCase):
    def setUp(self):
        """测试初始化"""
        self.solver = QuantumTunnelingSolver()
    
    def test_initial_conditions(self):
        """测试初始条件设置"""
        # 验证项目说明中要求的初始条件
        self.assertEqual(self.solver.Nx, 220, "空间点数不符合要求")
        self.assertEqual(self.solver.Nt, 300, "时间步数不符合要求")
        self.assertEqual(self.solver.x0, 40, "初始波包位置不符合要求")
        self.assertEqual(self.solver.k0, 0.5, "初始波包波数不符合要求")
    
    def test_potential_setup(self):
        """测试势垒设置"""
        # 验证项目说明中要求的势垒参数
        self.assertEqual(self.solver.barrier_width, 3, "势垒宽度不符合要求")
        self.assertEqual(self.solver.barrier_height, 1.0, "势垒高度不符合要求")
        
        # 验证势函数是否正确设置
        self.assertEqual(len(self.solver.V), self.solver.Nx, "势函数长度不匹配")
        self.assertTrue(np.all(self.solver.V[self.solver.Nx//2:self.solver.Nx//2+self.solver.barrier_width] == self.solver.barrier_height),
                        "势垒区域设置不正确")
    
    def test_solution_method(self):
        """测试求解方法"""
        # 验证项目说明中要求的Crank-Nicolson方法实现
        x, V, B, C = self.solver.solve_schrodinger()
        
        # 验证解的形状
        self.assertEqual(B.shape, (self.solver.Nx, self.solver.Nt), "解的形状不正确")
        self.assertEqual(C.shape, (self.solver.Nx, self.solver.Nt), "解的形状不正确")
        
        # 验证概率守恒
        initial_prob = np.sum(np.abs(B[:,0])**2)
        final_prob = np.sum(np.abs(B[:,-1])**2)
        self.assertAlmostEqual(initial_prob, final_prob, delta=0.01, 
                              msg="概率不守恒")
    
    def test_visualization(self):
        """测试可视化功能"""
        # 验证项目说明中要求的可视化功能
        try:
            self.solver.plot_evolution()
            visualization_works = True
        except:
            visualization_works = False
        
        self.assertTrue(visualization_works, "可视化功能异常")

if __name__ == "__main__":
    unittest.main()