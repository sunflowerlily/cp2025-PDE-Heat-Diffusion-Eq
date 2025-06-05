import sys
import os
import numpy as np
import pytest

# 添加父目录到系统路径以导入被测试模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from solution.simpson_integration_solution import f, trapezoidal, simpson
from simpson_integration import f, trapezoidal, simpson

class TestSimpsonIntegration:
    """测试Simpson积分法与梯形法则"""
    
    def test_f_function(self):
        """测试被积函数f(x) = x^4 - 2x + 1"""
        assert f(0) == 1
        assert f(1) == 0
        assert f(2) == 13
        assert np.isclose(f(0.5), 0.0625 - 1 + 1)
    
    def test_trapezoidal_integration(self):
        """测试梯形法则积分"""
        # 已知精确解为4.4
        result = trapezoidal(f, 0, 2, 1000)
        assert np.isclose(result, 4.4, rtol=1e-4)
        
    def test_simpson_integration(self):
        """测试Simpson法则积分"""
        # 已知精确解为4.4
        result = simpson(f, 0, 2, 1000)
        assert np.isclose(result, 4.4, rtol=1e-6)
        
    def test_simpson_odd_N(self):
        """测试当N为奇数时Simpson法则应抛出异常"""
        with pytest.raises(ValueError, match="Simpson 法则要求 N 必须为偶数"):
            simpson(f, 0, 2, 999)
            
    def test_methods_comparison(self):
        """比较两种方法的精度"""
        trapezoidal_result = trapezoidal(f, 0, 2, 100)
        simpson_result = simpson(f, 0, 2, 100)
        exact = 4.4
        
        trapezoidal_error = abs(trapezoidal_result - exact) / exact
        simpson_error = abs(simpson_result - exact) / exact
        
        # Simpson法则误差应小于梯形法则
        assert simpson_error < trapezoidal_error

if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__])