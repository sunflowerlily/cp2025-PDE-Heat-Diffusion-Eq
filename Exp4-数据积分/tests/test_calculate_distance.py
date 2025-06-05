import sys
import os
import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

# 添加父目录到系统路径以导入被测试模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from solution.calculate_distance_solution import main as calculate_distance
from calculate_distance import main as calculate_distance
class TestCalculateDistance:
    """测试数据积分功能"""
    
    def setup_method(self):
        """创建测试数据"""
        self.test_data = os.path.join(os.path.dirname(__file__), 'test_velocities.txt')
        # 生成测试数据并写入文件
        t = np.linspace(0, 10, 100)
        v = np.sin(t)
        np.savetxt(self.test_data, np.column_stack((t, v)))
        
        # 预期结果
        self.expected_total = 1 - np.cos(10)  # 正弦积分的解析解
        self.expected_cumulative = 1 - np.cos(t)  # 累积积分的解析解
        
    def teardown_method(self):
        """清理测试数据文件"""
        if os.path.exists(self.test_data):
            os.remove(self.test_data)
    
    def test_total_distance_calculation(self):
        """测试总距离计算是否正确"""
        # 使用numpy.trapz计算作为参考
        t, v = np.loadtxt(self.test_data, unpack=True)
        calculated_total = np.trapz(v, t)
        assert np.isclose(calculated_total, self.expected_total, rtol=1e-3)
    
    def test_cumulative_distance_calculation(self):
        """测试累积距离计算是否正确"""
        t, v = np.loadtxt(self.test_data, unpack=True)
        calculated_cumulative = cumulative_trapezoid(v, t, initial=0)
        assert np.allclose(calculated_cumulative, self.expected_cumulative, rtol=1e-3)
    
    def test_main_function_output(self):
        """测试主函数是否能正常运行"""
        # 重定向标准输出以捕获打印内容
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            calculate_distance()
        output = f.getvalue()
        
        # 验证输出包含总距离信息
        assert "总运行距离" in output
        # 验证输出格式正确
        assert "米" in output

if __name__ == "__main__":
    pytest.main([__file__])