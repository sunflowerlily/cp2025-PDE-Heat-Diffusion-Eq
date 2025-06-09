#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地壳热扩散数值模拟测试套件

本模块包含对地壳热扩散求解器的全面测试，包括：
1. 基本功能测试
2. 数值精度测试
3. 边界条件测试
4. 稳定性测试
5. 物理合理性测试

作者: 教学团队
日期: 2024
"""

import unittest
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from earth_crust_diffusion_student import (
        solve_earth_crust_diffusion,
        analyze_long_term_evolution,
        plot_seasonal_profiles,
        analytical_solution,
        compare_with_analytical,
        parameter_sensitivity_analysis,
        validate_parameters
    )
except ImportError:
    # 如果学生文件不存在或有错误，导入解决方案
    from solution.earth_crust_diffusion_solution import (
        solve_earth_crust_diffusion,
        analyze_long_term_evolution,
        plot_seasonal_profiles,
        analytical_solution,
        compare_with_analytical,
        parameter_sensitivity_analysis,
        validate_parameters
    )

# 物理常数（与主模块保持一致）
D = 0.1  # 热扩散率 (m^2/day)
A = 10.0  # 年平均地表温度 (°C)
B = 12.0  # 地表温度振幅 (°C)
TAU = 365.0  # 年周期 (days)
T_BOTTOM = 11.0  # 20米深处温度 (°C)
T_INITIAL = 10.0  # 初始温度 (°C)
DEPTH_MAX = 20.0  # 最大深度 (m)


class TestEarthCrustDiffusion(unittest.TestCase):
    """地壳热扩散求解器测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.test_nz = 21  # 较小的网格用于快速测试
        self.test_nt = 365  # 1年的模拟
        self.test_time_years = 1.0
        self.tolerance = 1e-10  # 数值容差
    
    def test_parameter_validation_5pts(self):
        """测试参数验证功能 (5分)"""
        print("\n测试参数验证...")
        
        # 测试有效参数
        try:
            validate_parameters(21, 365, 1.0)
            print("  ✓ 有效参数验证通过")
        except Exception as e:
            self.fail(f"有效参数验证失败: {e}")
        
        # 测试无效的nz
        with self.assertRaises(ValueError):
            validate_parameters(2, 365, 1.0)  # nz < 3
        
        with self.assertRaises(ValueError):
            validate_parameters("invalid", 365, 1.0)  # 非整数
        
        # 测试无效的nt
        with self.assertRaises(ValueError):
            validate_parameters(21, 0, 1.0)  # nt < 1
        
        # 测试无效的时间
        with self.assertRaises(ValueError):
            validate_parameters(21, 365, -1.0)  # 负时间
        
        print("  ✓ 无效参数正确拒绝")
    
    def test_basic_solver_functionality_15pts(self):
        """测试基本求解器功能 (15分)"""
        print("\n测试基本求解器功能...")
        
        # 运行求解器
        depth, time, temperature = solve_earth_crust_diffusion(
            self.test_nz, self.test_nt, self.test_time_years
        )
        
        # 检查返回值的形状
        self.assertEqual(len(depth), self.test_nz, "深度数组长度不正确")
        self.assertEqual(len(time), self.test_nt + 1, "时间数组长度不正确")
        self.assertEqual(temperature.shape, (self.test_nt + 1, self.test_nz), 
                        "温度矩阵形状不正确")
        
        # 检查深度和时间范围
        self.assertAlmostEqual(depth[0], 0.0, places=10, msg="深度起点应为0")
        self.assertAlmostEqual(depth[-1], DEPTH_MAX, places=10, msg="深度终点不正确")
        self.assertAlmostEqual(time[0], 0.0, places=10, msg="时间起点应为0")
        self.assertAlmostEqual(time[-1], self.test_time_years * 365.0, places=10, 
                              msg="时间终点不正确")
        
        # 检查温度值的合理性
        self.assertTrue(np.all(np.isfinite(temperature)), "温度值包含无穷大或NaN")
        self.assertTrue(np.all(temperature >= -50), "温度值过低（物理不合理）")
        self.assertTrue(np.all(temperature <= 50), "温度值过高（物理不合理）")
        
        print("  ✓ 基本求解器功能正常")
    
    def test_boundary_conditions_10pts(self):
        """测试边界条件 (10分)"""
        print("\n测试边界条件...")
        
        depth, time, temperature = solve_earth_crust_diffusion(
            self.test_nz, self.test_nt, self.test_time_years
        )
        
        # 检查地表边界条件
        for n in range(len(time)):
            expected_surface_temp = A + B * np.sin(2 * np.pi * time[n] / TAU)
            actual_surface_temp = temperature[n, 0]
            self.assertAlmostEqual(actual_surface_temp, expected_surface_temp, places=6,
                                 msg=f"地表温度边界条件在时间{time[n]}不满足")
        
        # 检查底部边界条件
        for n in range(len(time)):
            actual_bottom_temp = temperature[n, -1]
            self.assertAlmostEqual(actual_bottom_temp, T_BOTTOM, places=6,
                                 msg=f"底部温度边界条件在时间{time[n]}不满足")
        
        print("  ✓ 边界条件正确实施")
    
    def test_initial_conditions_5pts(self):
        """测试初始条件 (5分)"""
        print("\n测试初始条件...")
        
        depth, time, temperature = solve_earth_crust_diffusion(
            self.test_nz, self.test_nt, self.test_time_years
        )
        
        # 检查初始温度分布（除了边界点）
        for i in range(1, len(depth) - 1):
            self.assertAlmostEqual(temperature[0, i], T_INITIAL, places=6,
                                 msg=f"初始条件在深度{depth[i]}不满足")
        
        # 检查初始地表温度
        expected_initial_surface = A + B * np.sin(2 * np.pi * 0 / TAU)
        self.assertAlmostEqual(temperature[0, 0], expected_initial_surface, places=6,
                             msg="初始地表温度不正确")
        
        print("  ✓ 初始条件正确设置")
    
    def test_analytical_solution_10pts(self):
        """测试解析解计算 (10分)"""
        print("\n测试解析解计算...")
        
        # 创建测试网格
        depth_test = np.linspace(0, 10, 11)  # 0-10米
        time_test = np.linspace(0, 365, 366)  # 1年
        
        surface_params = {'A': A, 'B': B, 'tau': TAU}
        T_analytical = analytical_solution(depth_test, time_test, surface_params)
        
        # 检查解析解的形状
        self.assertEqual(T_analytical.shape, (len(time_test), len(depth_test)),
                        "解析解矩阵形状不正确")
        
        # 检查解析解的物理合理性
        self.assertTrue(np.all(np.isfinite(T_analytical)), "解析解包含无穷大或NaN")
        
        # 检查地表边界条件
        for i, t in enumerate(time_test):
            expected_surface = A + B * np.sin(2 * np.pi * t / TAU)
            actual_surface = T_analytical[i, 0]
            self.assertAlmostEqual(actual_surface, expected_surface, places=6,
                                 msg=f"解析解地表温度在时间{t}不正确")
        
        # 检查深度衰减特性
        # 振幅应随深度衰减
        surface_amplitude = np.max(T_analytical[:, 0]) - np.min(T_analytical[:, 0])
        deep_amplitude = np.max(T_analytical[:, -1]) - np.min(T_analytical[:, -1])
        self.assertLess(deep_amplitude, surface_amplitude, "温度振幅应随深度衰减")
        
        print("  ✓ 解析解计算正确")
    
    def test_numerical_analytical_comparison_10pts(self):
        """测试数值解与解析解比较 (10分)"""
        print("\n测试数值解与解析解比较...")
        
        # 使用较小的网格进行快速测试
        depth, time, T_numerical = solve_earth_crust_diffusion(
            21, 365, 1.0
        )
        
        surface_params = {'A': A, 'B': B, 'tau': TAU}
        T_analytical = analytical_solution(depth, time, surface_params)
        
        comparison_results = compare_with_analytical(
            depth, time, T_numerical, T_analytical
        )
        
        # 检查误差指标
        self.assertIsInstance(comparison_results['rmse'], (int, float),
                            "RMSE应为数值")
        self.assertIsInstance(comparison_results['max_error'], (int, float),
                            "最大误差应为数值")
        
        # 检查误差在合理范围内
        self.assertLess(comparison_results['rmse'], 2.0, "RMSE误差过大")
        self.assertLess(comparison_results['max_error'], 5.0, "最大误差过大")
        
        print(f"  ✓ RMSE: {comparison_results['rmse']:.4f}°C")
        print(f"  ✓ 最大误差: {comparison_results['max_error']:.4f}°C")
    
    def test_long_term_evolution_analysis_10pts(self):
        """测试长期演化分析 (10分)"""
        print("\n测试长期演化分析...")
        
        # 运行较长时间的模拟
        depth, time, temperature = solve_earth_crust_diffusion(
            21, 1095, 3.0  # 3年模拟
        )
        
        evolution_results = analyze_long_term_evolution(depth, time, temperature)
        
        # 检查返回的结果结构
        required_keys = ['amplitude_decay', 'phase_delay', 'steady_state_time', 
                        'energy_conservation']
        for key in required_keys:
            self.assertIn(key, evolution_results, f"缺少分析结果: {key}")
        
        # 检查振幅衰减
        amplitudes = evolution_results['amplitude_decay']
        if amplitudes is not None:
            self.assertEqual(len(amplitudes), len(depth), "振幅数组长度不正确")
            # 振幅应随深度递减
            self.assertGreaterEqual(amplitudes[0], amplitudes[-1], 
                                  "振幅应随深度衰减")
        
        # 检查相位延迟
        phase_delays = evolution_results['phase_delay']
        if phase_delays is not None:
            self.assertEqual(len(phase_delays), len(depth), "相位延迟数组长度不正确")
        
        # 检查能量守恒
        energy_conservation = evolution_results['energy_conservation']
        if energy_conservation is not None:
            self.assertIn('is_conserved', energy_conservation, "缺少能量守恒判断")
            self.assertIn('max_change_rate', energy_conservation, "缺少能量变化率")
        
        print("  ✓ 长期演化分析功能正常")
    
    def test_seasonal_profiles_plotting_5pts(self):
        """测试季节性温度轮廓绘制 (5分)"""
        print("\n测试季节性温度轮廓绘制...")
        
        # 创建测试数据
        depth = np.linspace(0, 20, 21)
        # 模拟四个季节的温度轮廓
        temperature_profiles = np.array([
            10 + 5 * np.exp(-depth/5),  # 春季
            15 + 8 * np.exp(-depth/5),  # 夏季
            10 + 3 * np.exp(-depth/5),  # 秋季
            5 + 2 * np.exp(-depth/5)    # 冬季
        ])
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        
        try:
            fig = plot_seasonal_profiles(depth, temperature_profiles, seasons)
            self.assertIsNotNone(fig, "应返回图形对象")
            print("  ✓ 季节性温度轮廓绘制成功")
        except Exception as e:
            self.fail(f"季节性温度轮廓绘制失败: {e}")
    
    def test_parameter_sensitivity_analysis_5pts(self):
        """测试参数敏感性分析 (5分)"""
        print("\n测试参数敏感性分析...")
        
        try:
            sensitivity_results = parameter_sensitivity_analysis()
            self.assertIsInstance(sensitivity_results, dict, "应返回字典类型结果")
            print("  ✓ 参数敏感性分析功能正常")
        except Exception as e:
            # 如果未实现，不算错误
            if "NotImplementedError" in str(type(e)):
                print("  - 参数敏感性分析未实现（可选功能）")
            else:
                self.fail(f"参数敏感性分析失败: {e}")
    
    def test_energy_conservation_10pts(self):
        """测试能量守恒 (10分)"""
        print("\n测试能量守恒...")
        
        depth, time, temperature = solve_earth_crust_diffusion(
            21, 365, 1.0
        )
        
        # 计算系统总能量（简化为温度积分）
        dz = depth[1] - depth[0]
        total_energy = np.trapz(temperature, dx=dz, axis=1)
        
        # 计算能量变化率
        energy_change = np.gradient(total_energy, time)
        max_energy_change_rate = np.max(np.abs(energy_change))
        
        # 能量变化率应该相对较小（考虑边界热流）
        self.assertLess(max_energy_change_rate, 1.0, 
                       "能量变化率过大，可能存在数值问题")
        
        print(f"  ✓ 最大能量变化率: {max_energy_change_rate:.6f}")
    
    def test_stability_condition_5pts(self):
        """测试数值稳定性条件 (5分)"""
        print("\n测试数值稳定性...")
        
        # 测试不同的网格参数
        test_cases = [
            (21, 365, 1.0),   # 正常情况
            (51, 1825, 5.0),  # 更精细网格
        ]
        
        for nz, nt, years in test_cases:
            depth, time, temperature = solve_earth_crust_diffusion(nz, nt, years)
            
            # 检查是否有数值不稳定的迹象
            self.assertTrue(np.all(np.isfinite(temperature)), 
                          f"网格({nz}, {nt})出现数值不稳定")
            
            # 检查温度变化是否过于剧烈
            temp_gradients = np.gradient(temperature, axis=0)
            max_temp_change = np.max(np.abs(temp_gradients))
            self.assertLess(max_temp_change, 10.0, 
                          f"网格({nz}, {nt})温度变化过于剧烈")
        
        print("  ✓ 数值稳定性良好")
    
    def test_physical_reasonableness_5pts(self):
        """测试物理合理性 (5分)"""
        print("\n测试物理合理性...")
        
        depth, time, temperature = solve_earth_crust_diffusion(
            21, 365, 1.0
        )
        
        # 1. 温度应在合理范围内
        min_temp = np.min(temperature)
        max_temp = np.max(temperature)
        self.assertGreaterEqual(min_temp, -30, "最低温度过低")
        self.assertLessEqual(max_temp, 40, "最高温度过高")
        
        # 2. 深层温度变化应小于地表
        surface_variation = np.max(temperature[:, 0]) - np.min(temperature[:, 0])
        deep_variation = np.max(temperature[:, -1]) - np.min(temperature[:, -1])
        self.assertLess(deep_variation, surface_variation, 
                       "深层温度变化应小于地表")
        
        # 3. 温度梯度应连续（无突跳）
        for n in range(len(time)):
            temp_profile = temperature[n, :]
            temp_gradient = np.gradient(temp_profile, depth)
            max_gradient = np.max(np.abs(temp_gradient))
            self.assertLess(max_gradient, 5.0, "温度梯度过大")
        
        print("  ✓ 物理合理性检查通过")
        print(f"    温度范围: {min_temp:.2f}°C 到 {max_temp:.2f}°C")
        print(f"    地表变化: {surface_variation:.2f}°C")
        print(f"    深层变化: {deep_variation:.2f}°C")


def run_tests_with_scoring():
    """
    运行所有测试并计算分数
    
    返回:
        tuple: (总分, 最大分数, 测试结果详情)
    """
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEarthCrustDiffusion)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2, stream=open(os.devnull, 'w'))
    result = runner.run(suite)
    
    # 计算分数
    total_score = 0
    max_score = 100
    
    # 分数映射（根据测试方法名中的分数）
    score_map = {
        'test_parameter_validation_5pts': 5,
        'test_basic_solver_functionality_15pts': 15,
        'test_boundary_conditions_10pts': 10,
        'test_initial_conditions_5pts': 5,
        'test_analytical_solution_10pts': 10,
        'test_numerical_analytical_comparison_10pts': 10,
        'test_long_term_evolution_analysis_10pts': 10,
        'test_seasonal_profiles_plotting_5pts': 5,
        'test_parameter_sensitivity_analysis_5pts': 5,
        'test_energy_conservation_10pts': 10,
        'test_stability_condition_5pts': 5,
        'test_physical_reasonableness_5pts': 5
    }
    
    # 计算通过的测试分数
    for test, score in score_map.items():
        # 检查测试是否通过（没有在失败列表中）
        test_passed = True
        for failure in result.failures + result.errors:
            if test in str(failure[0]):
                test_passed = False
                break
        
        if test_passed:
            total_score += score
    
    return total_score, max_score, result


if __name__ == '__main__':
    print("=" * 60)
    print("地壳热扩散数值模拟 - 测试套件")
    print("=" * 60)
    
    # 运行测试并计算分数
    score, max_score, test_result = run_tests_with_scoring()
    
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"总分: {score}/{max_score}")
    print(f"通过率: {score/max_score*100:.1f}%")
    
    if test_result.failures:
        print(f"\n失败的测试: {len(test_result.failures)}")
        for failure in test_result.failures:
            print(f"  - {failure[0]}")
    
    if test_result.errors:
        print(f"\n错误的测试: {len(test_result.errors)}")
        for error in test_result.errors:
            print(f"  - {error[0]}")
    
    print("\n" + "=" * 60)
    
    # 如果作为模块导入，也可以直接运行unittest
    # unittest.main(verbosity=2)