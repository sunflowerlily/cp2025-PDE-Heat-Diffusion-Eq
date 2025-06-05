import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import os

def main():
    try:
        # 1. 获取数据文件路径（使用绝对路径）
        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_file = os.path.join(data_dir, 'Velocities.txt')
        
        # 2. 读取数据
        data = np.loadtxt(data_file)
        t = data[:, 0]  # 时间列
        v = data[:, 1]  # 速度列

        # 2. 计算总距离
        total_distance = np.trapz(v, t)
        print(f"总运行距离: {total_distance:.2f} 米")

        # 3. 计算累积距离
        distance = cumulative_trapezoid(v, t, initial=0)

        # 4. 绘制图表
        plt.figure(figsize=(10, 6))
        
        # Plot velocity curve
        plt.plot(t, v, 'b-', label='Velocity (m/s)')
        
        # Plot distance curve 
        plt.plot(t, distance, 'r--', label='Distance (m)')
        
        # Chart decoration
        plt.title('Velocity and Distance vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s) / Distance (m)')
        plt.legend()
        plt.grid(True)
        
        # 显示图表
        plt.show()
    except FileNotFoundError:
        print(f"错误：找不到数据文件 {data_file}")
        print("请确保数据文件存在于项目目录中")

if __name__ == '__main__':
    main()