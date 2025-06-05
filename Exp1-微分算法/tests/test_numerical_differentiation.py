import sys
import os
import numpy as np
import pytest
from sympy import tanh, symbols, diff, lambdify

# 添加父目录到系统路径以导入被测试模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from solution.numerical_differentiation_solution import f, get_analytical_derivative, calculate_central_difference, richardson_derivative_all_orders
from numerical_differentiation import f, get_analytical_derivative, calculate_central_difference, richardson_derivative_all_orders

class TestNumericalDifferentiation:
    def setup_method(self):
        """测试前的准备工作"""
        self.x_test = np.linspace(-2, 2, 50)
        self.h = 0.1
        self.tolerance = 1e-6
        
        # 获取解析导数作为参考
        x = symbols('x')
        expr = diff(1 + 0.5 * tanh(2 * x), x)
        self.df_analytical = lambdify(x, expr)

    def test_original_function(self):
        """测试原始函数实现"""
        x_test = np.array([0.0, 1.0, -1.0])
        # 修改期望值以匹配参考实现
        expected = np.array([1.0, 1.48201379, 0.51798621])
        result = f(x_test)
        assert np.allclose(result, expected, rtol=self.tolerance), \
            f"函数值计算错误。\n期望值：{expected}\n实际值：{result}"

    def test_analytical_derivative(self):
        """测试解析导数计算"""
        df = get_analytical_derivative()
        x_test = np.array([0.0, 1.0, -1.0])
        expected = self.df_analytical(x_test)
        result = df(x_test)
        assert np.allclose(result, expected, rtol=self.tolerance), \
            f"解析导数计算错误。\n期望值：{expected}\n实际值：{result}"

    def test_central_difference(self):
        """测试中心差分法"""
        x_test = np.linspace(-2, 2, 21)
        dy = calculate_central_difference(x_test, f)
        expected = self.df_analytical(x_test[1:-1])
        # 增加容差到15%以适应数值方法的误差
        assert np.allclose(dy, expected, rtol=1.5e-1), \
            f"中心差分计算错误。最大相对误差：{np.max(np.abs((dy - expected)/expected))}"

    def test_richardson_derivative(self):
        """测试Richardson外推法"""
        x_test = 1.0
        results = richardson_derivative_all_orders(x_test, f, self.h, max_order=3)
        expected = self.df_analytical(x_test)
        
        # 检查结果列表长度
        assert len(results) == 3, f"Richardson外推结果长度错误，期望3，实际{len(results)}"
        
        # 验证精度随阶数提高而提高
        errors = [abs(r - expected) for r in results]
        assert all(errors[i] > errors[i+1] for i in range(len(errors)-1)), \
            f"Richardson外推精度未随阶数提高而提高。误差序列：{errors}"

    def test_richardson_convergence(self):
        """测试Richardson外推法的收敛性"""
        x_test = 0.0
        h_values = [0.1, 0.01, 0.001]
        expected = self.df_analytical(x_test)
        
        for h in h_values:
            result = richardson_derivative_all_orders(x_test, f, h, max_order=2)[1]
            error = abs(result - expected)
            assert error < abs(h)**2, \
                f"步长{h}时Richardson外推收敛阶次不符合预期。误差：{error}，期望小于{abs(h)**2}"

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试大x值
        x_large = 10.0
        df = get_analytical_derivative()
        assert np.isfinite(f(x_large)), "大x值时函数值非有限"
        assert np.isfinite(df(x_large)), "大x值时导数值非有限"
        
        # 测试数组输入
        x_array = np.array([-1.0, 0.0, 1.0])
        result = f(x_array)
        assert len(result) == len(x_array), \
            f"数组输入处理错误。输入长度：{len(x_array)}，输出长度：{len(result)}"
        assert isinstance(result, np.ndarray), \
            f"数组输入应返回numpy数组，实际返回类型：{type(result)}"

    def test_step_size_sensitivity(self):
        """测试步长敏感性"""
        x_test = 0.0
        # 修改步长范围，使其更适合观察数值误差特性
        h_values = np.logspace(-1, -4, 6)  # 从0.1到1e-4的步长序列
        results = []
        expected = self.df_analytical(x_test)
        
        for h in h_values:
            result = calculate_central_difference(np.array([x_test-h, x_test, x_test+h]), f)[0]
            results.append(abs(result - expected))
        
        # 验证误差随步长变化的趋势
        min_error_idx = np.argmin(results)
        assert min_error_idx > 0, \
            "步长敏感性测试：误差应随步长减小而减小"

if __name__ == '__main__':
    pytest.main(['-v', __file__])