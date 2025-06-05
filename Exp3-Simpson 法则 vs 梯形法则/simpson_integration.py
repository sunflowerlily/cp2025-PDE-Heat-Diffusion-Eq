import numpy as np

# 待积分函数（学生需自行定义）
def f(x):
    # TODO: 实现被积函数 f(x) = x^4 - 2x + 1
    pass

# 梯形法则积分函数（供参考比较用）
def trapezoidal(f, a, b, N):
    """
    梯形法数值积分
    :param f: 被积函数
    :param a: 积分下限
    :param b: 积分上限  
    :param N: 子区间数
    :return: 积分近似值
    """
    # TODO: 实现梯形法则积分
    pass

# Simpson法则积分函数（学生需完成）
def simpson(f, a, b, N):
    """
    Simpson法数值积分
    :param f: 被积函数
    :param a: 积分下限
    :param b: 积分上限
    :param N: 子区间数（必须为偶数）
    :return: 积分近似值
    """
    # TODO: 实现Simpson法则积分
    # 注意：需先检查N是否为偶数
    pass

def main():
    a, b = 0, 2  # 积分区间
    exact_integral = 4.4  # 精确解

    for N in [100, 1000]:  # 不同子区间数
        # TODO: 调用积分函数并计算误差
        trapezoidal_result = None
        simpson_result = None
        
        # TODO: 计算相对误差
        trapezoidal_error = None  
        simpson_error = None

        # 输出结果（模板已给出）
        print(f"N = {N}")
        print(f"梯形法则结果: {trapezoidal_result:.8f}, 相对误差: {trapezoidal_error:.2e}")
        print(f"Simpson法则结果: {simpson_result:.8f}, 相对误差: {simpson_error:.2e}")
        print("-" * 40)

if __name__ == '__main__':
    main()