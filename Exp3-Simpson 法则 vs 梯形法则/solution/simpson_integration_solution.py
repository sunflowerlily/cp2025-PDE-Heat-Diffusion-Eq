import numpy as np

# 待积分函数
def f(x):
    return x**4 - 2*x + 1

# 已给出的梯形法则积分函数，供参考比较用
def trapezoidal(f, a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    fx = f(x)
    integral = h * (0.5 * fx[0] + np.sum(fx[1:-1]) + 0.5 * fx[-1])
    return integral

# Simpson 法则积分函数
def simpson(f, a, b, N):
    if N % 2 != 0:
        raise ValueError("Simpson 法则要求 N 必须为偶数")
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    fx = f(x)
    # 奇数索引（1,3,5,...,N-1）
    odd_sum = np.sum(fx[1:N:2])
    # 偶数索引（2,4,6,...,N-2）
    even_sum = np.sum(fx[2:N:2])
    integral = h / 3 * (fx[0] + 4 * odd_sum + 2 * even_sum + fx[N])
    return integral

def main():
    a, b = 0, 2
    exact_integral = 4.4

    for N in [100, 1000]:
        trapezoidal_result = trapezoidal(f, a, b, N)
        simpson_result = simpson(f, a, b, N)

        trapezoidal_error = abs(trapezoidal_result - exact_integral) / exact_integral
        simpson_error = abs(simpson_result - exact_integral) / exact_integral

        print(f"N = {N}")
        print(f"梯形法则结果: {trapezoidal_result:.8f}, 相对误差: {trapezoidal_error:.2e}")
        print(f"Simpson法则结果: {simpson_result:.8f}, 相对误差: {simpson_error:.2e}")
        print("-" * 40)

if __name__ == '__main__':
    main()