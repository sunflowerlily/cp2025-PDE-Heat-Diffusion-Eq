import sys
import os
import numpy as np
import pytest

# 添加父目录到系统路径以导入被测试模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from solution.rate_sensitivity_solution import q3a
from rate_sensitivity import q3a

class TestRateSensitivity:
    """测试3-α反应速率计算和温度敏感性指数"""
    
    def test_q3a_zero_temperature(self):
        """测试零温度和负温度的处理"""
        assert q3a(0.0) == 0.0
        assert q3a(-1.0e8) == 0.0
    
    def test_q3a_normal_temperature(self):
        """测试正常温度范围的反应速率计算"""
        # 测试一些典型温度点
        test_temps = [1.0e8, 2.5e8, 5.0e8, 1.0e9]
        expected_rates = [
            3.8551e-08,  # at 1.0e8 K
            7.3219e+02,  # at 2.5e8 K
            6.1048e+05,  # at 5.0e8 K
            6.2323e+06   # at 1.0e9 K (updated from 1.1482e-03)
        ]
        
        for T, expected in zip(test_temps, expected_rates):
            calculated = q3a(T)
            # 由于指数计算的数值敏感性，使用相对误差进行比较
            assert np.abs(calculated - expected) / expected < 1e-3, \
                f"温度 {T} K 处的计算结果与预期值相差过大"
    
    def test_temperature_sensitivity(self):
        """测试温度敏感性指数的计算"""
        T0 = 1.0e8  # 测试温度点
        h = 1.0e-8  # 扰动因子
        
        # 手动计算温度敏感性指数
        q_T0 = q3a(T0)
        q_T0_plus_dT = q3a(T0 * (1 + h))
        
        # 使用前向差分计算 nu
        nu = (T0 / q_T0) * (q_T0_plus_dT - q_T0) / (h * T0)
        
        # 在 T = 10^8 K 时，nu 应该约为 41
        assert 40 < nu < 42, \
            f"T = 10^8 K 时的温度敏感性指数计算错误，期望约41，得到{nu}"
    
    def test_q3a_formula(self):
        """测试反应速率公式的基本结构"""
        T = 1.0e8
        T8 = T / 1.0e8
        
        # 手动计算预期值
        expected = 5.09e11 * T8**(-3.0) * np.exp(-44.027/T8)
        calculated = q3a(T)
        
        # 检查计算结果是否符合公式结构
        assert np.abs(calculated - expected) / expected < 1e-10, \
            "反应速率计算公式结构不正确"


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__])