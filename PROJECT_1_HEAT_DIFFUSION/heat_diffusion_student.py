"""
学生模板：PROJECT_1_HEAT_DIFFUSION (铝棒热传导问题)
文件：heat_diffusion_student.py

重要：
1.  函数名称、参数名称和类型、返回值类型必须与 `solution/heat_diffusion_solution.py` 中的参考答案保持一致。
2.  请在标记为 `# TODO:` 的部分填写你的代码。
3.  你可以自由添加辅助函数，但不要修改主要函数的签名。
4.  完成代码后，请撰写实验报告。
"""
import numpy as np
import matplotlib.pyplot as plt

# 默认物理参数 (参考自 solution/heat_diffusion_solution.py)
DEFAULT_K = 237.0      # 导热系数 W/(m·K)
DEFAULT_C = 900.0      # 比热容 J/(kg·K)
DEFAULT_RHO = 2700.0   # 密度 kg/m³
DEFAULT_L = 1.0        # 棒长 m
DEFAULT_T0 = 100.0     # 默认初始温度 K

class HeatDiffusionSolver:
    """
    (学生版) 求解一维热传导方程。
    方程: ∂T/∂t = D * ∂²T/∂x² - h_coeff * (T - T_env)
    其中 D 是热扩散率, h_coeff 是冷却系数, T_env 是环境温度。
    """

    def __init__(self, L=DEFAULT_L, K=DEFAULT_K, C=DEFAULT_C, rho=DEFAULT_RHO, 
                 nx=101, total_time=1000.0, dt=0.5, 
                 initial_condition_config=None,
                 boundary_conditions=((0,0), (0,0))):
        """
        初始化求解器。

        参数:
            L (float): 棒长 (m)。
            K (float): 导热系数 (W/(m·K))。
            C (float): 比热容 (J/(kg·K))。
            rho (float): 密度 (kg/m³)。
            nx (int): 空间网格点数。
            total_time (float): 总模拟时间 (s)。
            dt (float): 时间步长 (s)。
            initial_condition_config (float or dict): 初始条件配置。
                - float: 均匀初始温度 T0。
                - dict: 复杂初始条件, 例如: {'type': 'two_rods', 'T1': 100, 'T2': 50, 'split_x': 0.5}
                        如果为 None，则默认为 DEFAULT_T0。
            boundary_conditions (tuple): 边界条件。((类型左, 值左), (类型右, 值右))。
                                         类型 0 表示狄利克雷边界条件。
                                         例如: ((0, 0), (0, 0)) 表示 T(0,t)=0 和 T(L,t)=0。
        
        物理背景: 研究一维杆中的热量如何随时间和空间分布。
        数值方法: 采用显式有限差分法 (FTCS) 进行离散化求解。
        
        实现步骤:
        1.  根据输入参数计算热扩散率 D。
        2.  初始化空间网格 x (包含 nx 个点，总长度为 L)。
        3.  计算空间步长 dx。
        4.  初始化时间网格 t (总时间 total_time，时间步长 dt，计算总步数 nt)。
        5.  计算稳定性参数 r = D * dt / dx²。
        6.  存储初始条件配置和边界条件配置。
        7.  (可选) 打印计算出的参数 D, dx, dt, r，并检查稳定性条件 r <= 0.5 (无冷却时)。
        """
        self.L = L
        self.K = K
        self.C = C
        self.rho = rho
        self.nx = nx
        self.total_time = total_time
        self.dt = dt
        self.initial_condition_config = initial_condition_config if initial_condition_config is not None else DEFAULT_T0
        self.boundary_conditions = boundary_conditions

        # TODO: 计算热扩散率 D (self.D)
        # self.D = ...

        # TODO: 初始化空间网格 self.x 和计算空间步长 self.dx
        # self.dx = ...
        # self.x = ...

        # TODO: 计算总时间步数 self.nt 和时间网格 self.t
        # self.nt = ...
        # self.t = ...

        # TODO: 计算稳定性参数 r (self.r)
        # self.r = ...
        
        # 打印参数 (可选，但有助于调试)
        # print(f"物理参数: L={L:.2f}m, K={K:.1f}, C={C:.1f}, rho={rho:.1f}, D={self.D:.2e} m^2/s")
        # print(f"网格参数: nx={nx}, dx={self.dx:.4f}m, nt={self.nt}, dt={self.dt:.4f}s, total_time={total_time:.1f}s")
        # print(f"稳定性参数 r = D*dt/dx^2 = {self.r:.4f}")
        # if self.r > 0.5:
        #     print(f"警告: r = {self.r:.4f} > 0.5. 对于纯 FTCS 方法，解可能不稳定。")
        raise NotImplementedError(f"请在 {__file__} 中实现 HeatDiffusionSolver.__init__ 方法")

    def set_initial_condition(self):
        """
        根据 self.initial_condition_config 设置初始温度分布 T(x, 0)。

        返回:
            numpy.ndarray: 初始温度分布，形状为 (nx,)。
        
        实现提示:
        1. 创建一个长度为 self.nx 的零数组 T_initial。
        2. 如果 self.initial_condition_config 是一个数字 (int 或 float)，则将 T_initial 的所有元素设置为该值。
        3. 如果 self.initial_condition_config 是一个字典:
           a. 获取 'type' 字段。
           b. 如果 'type' 是 'two_rods'，则根据 'T1', 'T2', 'split_x' (中点) 设置 T_initial。
              split_x 表示分隔点的 x 坐标值。
           c. 如果 'type' 未知或配置无效，可以打印警告并使用默认值 DEFAULT_T0。
        4. 根据 self.boundary_conditions 中的狄利克雷条件更新 T_initial 的首尾元素。
           (例如，如果左边界是狄利克雷条件 (类型0)，则 T_initial[0] = 左边界值)
        """
        # TODO: 实现设置初始条件
        raise NotImplementedError(f"请在 {__file__} 中实现 HeatDiffusionSolver.set_initial_condition 方法")

    def explicit_finite_difference(self, h_coeff=0.0, T_env=0.0):
        """
        使用显式有限差分法 (FTCS) 求解一维热传导方程。
        可以包含牛顿冷却项。

        方程: T_i^{j+1} = T_i^j + r * (T_{i+1}^j - 2*T_i^j + T_{i-1}^j) - h_coeff * dt * (T_i^j - T_env)
        其中 r = self.D * self.dt / (self.dx**2)

        参数:
            h_coeff (float): 牛顿冷却的传热系数 (s^-1)。如果为0，则无冷却项。
            T_env (float): 牛顿冷却的环境温度 (K)。
            
        返回:
            tuple: (time_array, temperature_matrix)
                   time_array (np.ndarray): 时间点数组，形状 (nt+1,)。
                   temperature_matrix (np.ndarray): 每个网格点和时间步的温度 T(x,t)，形状 (nx, nt+1)。
        
        实现提示:
        1. (可选) 进行稳定性检查。对于 h_coeff > 0，近似稳定性条件为 1 - 2*r - h_coeff*self.dt >= 0。
           如果纯 FTCS (h_coeff=0)，则 r <= 0.5。
        2. 初始化温度矩阵 T_matrix，形状为 (self.nx, self.nt + 1)。
        3. 使用 self.set_initial_condition() 设置 T_matrix 的第一列 (t=0 时刻)。
        4. 进行时间步进循环 (从 j=0 到 self.nt-1):
           a. 内部节点更新 (从 i=1 到 self.nx-2):
              i.  计算扩散项: self.r * (T_matrix[i+1, j] - 2*T_matrix[i, j] + T_matrix[i-1, j])
              ii. 计算冷却项 (如果 h_coeff > 0): h_coeff * self.dt * (T_matrix[i, j] - T_env)
              iii.更新 T_matrix[i, j+1]
           b. 在每个时间步应用边界条件 (更新 T_matrix[0, j+1] 和 T_matrix[-1, j+1])。
              - 如果边界是狄利克雷类型 (类型0)，则直接赋值。
              - (高级) 将来可以扩展以支持诺伊曼或罗宾边界条件。
        """
        # TODO: 实现显式有限差分法
        raise NotImplementedError(f"请在 {__file__} 中实现 HeatDiffusionSolver.explicit_finite_difference 方法")

    def analytical_solution(self, T0_analytical=None, n_terms=100):
        """
        计算特定条件下的一维热传导方程的解析解。
        条件: T(0,t) = 0, T(L,t) = 0, 初始条件 T(x,0) = T0 (常数)。
        此解析解仅对这些特定边界和初始条件有效。

        公式: T(x,t) = sum_{n=1,3,5,...}^{inf} (4*T0)/(n*pi) * sin(n*pi*x/L) * exp(-(n*pi/L)^2 * D * t)

        参数:
            T0_analytical (float, optional): 用于解析解的初始均匀温度 T0。
                                             如果未提供且 self.initial_condition_config 不是简单的浮点数，则默认为 DEFAULT_T0。
            n_terms (int): 傅里叶级数中求和的项数。
            
        返回:
            tuple: (time_array, temperature_matrix_analytical)
                   time_array (np.ndarray): 时间点数组。
                   temperature_matrix_analytical (np.ndarray): 解析解温度 T(x,t)。
        
        实现提示:
        1. 确定用于计算的 T0_analytical。如果参数未提供，则尝试从 self.initial_condition_config 获取；
           如果 self.initial_condition_config 复杂，则使用 DEFAULT_T0 并打印警告。
        2. 初始化解析解矩阵 T_an，形状为 (self.nx, self.nt + 1)。
        3. 使用 np.meshgrid 创建空间和时间网格 X_grid, Time_grid (注意索引方式 'ij')。
        4. 循环 n_odd 从 1 到 2*n_terms-1，步长为 2 (即 n = 1, 3, 5, ...):
           a. 计算波数 kn = n_odd * np.pi / self.L。
           b. 计算系数 term_coeff = (4 * T0_analytical) / (n_odd * np.pi)。
           c. 计算空间部分 term_spatial = np.sin(kn * X_grid)。
           d. 计算时间部分 term_temporal = np.exp(-(kn**2) * self.D * Time_grid)。
           e. 将 term_coeff * term_spatial * term_temporal 加到 T_an 上。
        5. (可选但推荐) 确保解析解满足边界条件，特别是当边界条件非零时。
           对于此处的 T(0,t)=0, T(L,t)=0，级数本身应收敛到这些值。
        """
        # TODO: 实现解析解的计算
        raise NotImplementedError(f"请在 {__file__} 中实现 HeatDiffusionSolver.analytical_solution 方法")

    # --- 以下为绘图辅助函数，学生无需修改，可以直接使用 ---
    def plot_evolution(self, t_array, T_matrix, title="温度演化", save_fig=False, filename_prefix="student_plot_evo"):
        """
        绘制不同时间点的温度分布曲线。
        """
        plt.figure(figsize=(10, 6))
        num_plots = 5
        plot_indices = np.linspace(0, self.nt, num_plots, dtype=int)
        for k, j in enumerate(plot_indices):
            plt.plot(self.x, T_matrix[:, j], label=f't = {self.t[j]:.2f} s')
        plt.xlabel('位置 x (m)')
        plt.ylabel('温度 T (K)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if save_fig:
            plt.savefig(f"{filename_prefix}.png")
            print(f"图像已保存到 {filename_prefix}.png")
        plt.show()

    def plot_comparison_at_times(self, t_num, T_num, T_an, time_indices, title="数值解 vs 解析解", save_fig=False, filename_prefix="student_plot_comp"):
        """
        在指定时间点比较数值解和解析解。
        time_indices: 要比较的时间步的索引列表。
        """
        plt.figure(figsize=(12, 8))
        num_plots = len(time_indices)
        for i, time_idx in enumerate(time_indices):
            plt.subplot( (num_plots + 1) // 2, 2, i + 1)
            plt.plot(self.x, T_num[:, time_idx], 'o-', label=f'数值解 t={t_num[time_idx]:.1f}s', markersize=3, linewidth=1)
            plt.plot(self.x, T_an[:, time_idx], 'r--', label=f'解析解 t={t_num[time_idx]:.1f}s', linewidth=1)
            plt.xlabel('x (m)')
            plt.ylabel('T (K)')
            plt.title(f't = {t_num[time_idx]:.1f}s')
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        if save_fig:
            plt.savefig(f"{filename_prefix}.png")
            print(f"图像已保存到 {filename_prefix}.png")
        plt.show()

# --- 以下为根据 `项目说明.md` 中定义的具体问题/任务的求解函数框架 --- 
# --- 学生需要实现这些函数 --- 

def solve_heat_diffusion_problem_1(params, save_plot=False, plot_filename="problem1_student.png"):
    """
    (学生版) 任务1：基本模拟 - 求解铝棒中的热传导问题 (特定参数)。
    根据 `项目说明.md` 中的任务1，使用给定的物理和数值参数，
    模拟一根两端保持在0K，初始温度为100K的铝棒的温度随时间的变化。

    参数:
        params (dict): 包含以下键值对的字典:
            'L': float, 棒长 (m)
            'nx': int, 空间网格点数
            'total_time': float, 总模拟时间 (s)
            'dt': float, 时间步长 (s)
            'K': float, 导热系数 (W/(m·K))
            'C': float, 比热容 (J/(kg·K))
            'rho': float, 密度 (kg/m³)
            'T0': float, 初始温度 (K)
            'bc_left_val': float, 左边界温度 (K)
            'bc_right_val': float, 右边界温度 (K)
            'plot_times': list of float, 需要绘制温度分布图的时间点 (s)
        save_plot (bool): 是否保存图像。
        plot_filename (str): 保存图像的文件名。

    返回:
        tuple: (solver, t_array, T_matrix)
               solver (HeatDiffusionSolver): 使用的求解器实例。
               t_array (np.ndarray): 时间数组。
               T_matrix (np.ndarray): 温度矩阵。

    实现要求:
    1.  从 params 字典中提取必要的参数来实例化 HeatDiffusionSolver。
        - initial_condition_config 应为 params['T0']。
        - boundary_conditions 应为 ((0, params['bc_left_val']), (0, params['bc_right_val']))。
    2.  调用求解器的 explicit_finite_difference 方法得到数值解。
    3.  如果 save_plot 为 True，则绘制指定时间点 (params['plot_times']) 的温度分布图。
        - 你需要找到与 params['plot_times'] 中每个时间点最接近的时间步索引。
        - 可以使用 solver.plot_evolution，但它绘制的是固定数量的、均匀间隔的时间点。
          或者，你可以自己编写绘图逻辑，或者修改 plot_evolution 以接受特定的时间点/索引。
          一个更简单的方法是，如果 plot_times 列表不长，可以多次调用 plt.plot。
          或者，直接使用 solver.plot_evolution 并接受其默认的时间点选择，如果这对于任务来说足够的话。
          对于本作业，建议自行根据 plot_times 绘制，以精确匹配要求。
    """
    # TODO: 提取参数并实例化 HeatDiffusionSolver
    # solver = HeatDiffusionSolver(...)
    
    # TODO: 运行显式有限差分法
    # t_array, T_matrix = solver.explicit_finite_difference()

    # TODO: 如果 save_plot 为 True，则绘图
    # if save_plot:
    #     plt.figure(figsize=(10, 6))
    #     for t_plot in params['plot_times']:
    #         # 找到最接近 t_plot 的时间索引
    #         time_idx = np.argmin(np.abs(t_array - t_plot))
    #         plt.plot(solver.x, T_matrix[:, time_idx], label=f't = {t_array[time_idx]:.1f}s (目标 {t_plot}s)')
    #     plt.xlabel('位置 x (m)')
    #     plt.ylabel('温度 T (K)')
    #     plt.title('任务1：铝棒温度分布 (学生版)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(plot_filename)
    #     print(f"图像已保存到 {plot_filename}")
    #     plt.show()

    # return solver, t_array, T_matrix
    raise NotImplementedError(f"请在 {__file__} 中实现 solve_heat_diffusion_problem_1 函数")

def solve_heat_diffusion_analytical_comparison(params, n_terms=100, save_plot=False, plot_filename="analytical_comp_student.png"):
    """
    (学生版) 任务2：与解析解比较。
    针对T(x,0)=T0，T(0,t)=0，T(L,t)=0 的情况，比较数值解和解析解。

    参数:
        params (dict): 与 solve_heat_diffusion_problem_1 相同的参数字典。
                       确保 T0, bc_left_val, bc_right_val 适合解析解的条件。
        n_terms (int): 解析解中傅里叶级数的项数。
        save_plot (bool): 是否保存图像。
        plot_filename (str): 保存图像的文件名。

    返回:
        tuple: (solver, t_num, T_num, T_an)
               solver (HeatDiffusionSolver): 求解器实例。
               t_num (np.ndarray): 数值解的时间数组。
               T_num (np.ndarray): 数值解的温度矩阵。
               T_an (np.ndarray): 解析解的温度矩阵。

    实现要求:
    1.  实例化 HeatDiffusionSolver (确保初始条件和边界条件与解析解的假设一致，即初始温度均匀 T0，两端温度为0)。
    2.  计算数值解 (explicit_finite_difference)。
    3.  计算解析解 (analytical_solution)，使用 params['T0'] 作为 T0_analytical。
    4.  如果 save_plot 为 True，则使用 solver.plot_comparison_at_times 在 params['plot_times'] 指定的时间点比较数值解和解析解。
    """
    # TODO: 实例化求解器，计算数值解和解析解
    # solver = HeatDiffusionSolver(...)
    # t_num, T_num = solver.explicit_finite_difference()
    # _, T_an = solver.analytical_solution(T0_analytical=params['T0'], n_terms=n_terms)

    # TODO: 如果 save_plot 为 True，则绘图
    # if save_plot:
    #     time_indices_to_plot = []
    #     for t_val in params['plot_times']:
    #         time_indices_to_plot.append(np.argmin(np.abs(t_num - t_val)))
    #     solver.plot_comparison_at_times(t_num, T_num, T_an, time_indices_to_plot,
    #                                     title='任务2：数值解 vs 解析解 (学生版)',
    #                                     save_fig=True, filename_prefix=plot_filename.replace('.png',''))
    # return solver, t_num, T_num, T_an
    raise NotImplementedError(f"请在 {__file__} 中实现 solve_heat_diffusion_analytical_comparison 函数")

def analyze_stability(params_stable, params_unstable, save_plot=False, filename_stable="stable_student.png", filename_unstable="unstable_student.png"):
    """
    (学生版) 任务3：稳定性分析。
    通过比较稳定 (r <= 0.5) 和不稳定 (r > 0.5) 情况下的数值解来说明稳定性条件的重要性。

    参数:
        params_stable (dict): 稳定情况的参数字典 (类似 solve_heat_diffusion_problem_1 中的 params)。
        params_unstable (dict): 不稳定情况的参数字典 (通常只有 dt 不同，导致 r > 0.5)。
        save_plot (bool): 是否保存图像。
        filename_stable (str): 稳定情况图像文件名。
        filename_unstable (str): 不稳定情况图像文件名。

    返回:
        tuple: (solver_stable, T_stable, solver_unstable, T_unstable)

    实现要求:
    1.  使用 params_stable 实例化一个 HeatDiffusionSolver，计算其解，并打印其 r 值。
    2.  如果 save_plot 为 True，使用 solver.plot_evolution 绘制稳定情况的温度演化图。
    3.  使用 params_unstable 实例化另一个 HeatDiffusionSolver，计算其解，并打印其 r 值。
    4.  如果 save_plot 为 True，使用 solver.plot_evolution 绘制不稳定情况的温度演化图。
    """
    # TODO: 实现稳定情况的模拟和绘图
    # print("稳定情况分析:")
    # solver_stable = HeatDiffusionSolver(...)
    # print(f"稳定情况 r = {solver_stable.r:.4f}")
    # _, T_stable = solver_stable.explicit_finite_difference()
    # if save_plot:
    #     solver_stable.plot_evolution(_, T_stable, title='任务3：稳定数值解 (学生版)', save_fig=True, filename_prefix=filename_stable.replace('.png',''))

    # TODO: 实现不稳定情况的模拟和绘图
    # print("\n不稳定情况分析:")
    # solver_unstable = HeatDiffusionSolver(...)
    # print(f"不稳定情况 r = {solver_unstable.r:.4f}")
    # _, T_unstable = solver_unstable.explicit_finite_difference()
    # if save_plot:
    #     solver_unstable.plot_evolution(_, T_unstable, title='任务3：不稳定数值解 (学生版)', save_fig=True, filename_prefix=filename_unstable.replace('.png',''))

    # return solver_stable, T_stable, solver_unstable, T_unstable
    raise NotImplementedError(f"请在 {__file__} 中实现 analyze_stability 函数")

def solve_heat_diffusion_two_rods(params, save_plot=False, plot_filename="two_rods_student.png"):
    """
    (学生版) 任务4：不同初始条件 - 两根不同初始温度的铝棒接触。
    模拟两段初始温度不同的铝棒接触后的温度变化。

    参数:
        params (dict): 包含以下键值对的字典:
            'L', 'nx', 'total_time', 'dt', 'K', 'C', 'rho' (同上)
            'T1': float, 左半部分铝棒的初始温度 (K)
            'T2': float, 右半部分铝棒的初始温度 (K)
            'split_x_ratio': float, 分割点在总长度的比例 (例如 0.5 表示中点)
            'bc_left_val', 'bc_right_val': 左右边界温度 (K)
            'plot_times': list of float, 需要绘制温度分布图的时间点 (s)
        save_plot (bool): 是否保存图像。
        plot_filename (str): 保存图像的文件名。

    返回:
        tuple: (solver, t_array, T_matrix)

    实现要求:
    1.  构造 initial_condition_config 字典，类型为 'two_rods'，包含 T1, T2, 和 split_x (L * split_x_ratio)。
    2.  实例化 HeatDiffusionSolver。
    3.  计算数值解。
    4.  如果 save_plot 为 True，则在 params['plot_times'] 指定的时间点绘制温度分布图。
    """
    # TODO: 构造 initial_condition_config
    # initial_config = {
    #     'type': 'two_rods',
    #     'T1': params['T1'],
    #     'T2': params['T2'],
    #     'split_x': params['L'] * params.get('split_x_ratio', 0.5) # 默认为中点
    # }
    # boundary_config = ((0, params['bc_left_val']), (0, params['bc_right_val']))

    # TODO: 实例化求解器并计算
    # solver = HeatDiffusionSolver(L=params['L'], K=params['K'], C=params['C'], rho=params['rho'],
    #                            nx=params['nx'], total_time=params['total_time'], dt=params['dt'],
    #                            initial_condition_config=initial_config,
    #                            boundary_conditions=boundary_config)
    # t_array, T_matrix = solver.explicit_finite_difference()

    # TODO: 如果 save_plot 为 True，则绘图 (类似 problem_1)
    # if save_plot:
    #     plt.figure(figsize=(10, 6))
    #     for t_plot in params['plot_times']:
    #         time_idx = np.argmin(np.abs(t_array - t_plot))
    #         plt.plot(solver.x, T_matrix[:, time_idx], label=f't = {t_array[time_idx]:.1f}s (目标 {t_plot}s)')
    #     plt.xlabel('位置 x (m)')
    #     plt.ylabel('温度 T (K)')
    #     plt.title('任务4：两段不同初始温度铝棒 (学生版)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(plot_filename)
    #     print(f"图像已保存到 {plot_filename}")
    #     plt.show()

    # return solver, t_array, T_matrix
    raise NotImplementedError(f"请在 {__file__} 中实现 solve_heat_diffusion_two_rods 函数")

def solve_heat_diffusion_newton_cooling(params, save_plot=False, plot_filename="newton_cooling_student.png"):
    """
    (学生版) 任务5：考虑牛顿冷却定律。
    模拟在考虑牛顿冷却定律（即杆通过其表面与环境进行热交换）时铝棒的温度分布。

    参数:
        params (dict): 包含以下键值对的字典:
            'L', 'nx', 'total_time', 'dt', 'K', 'C', 'rho', 'T0',
            'bc_left_val', 'bc_right_val', 'plot_times' (同上)
            'h_coeff': float, 传热系数 (s^-1)
            'T_env': float, 环境温度 (K)
        save_plot (bool): 是否保存图像。
        plot_filename (str): 保存图像的文件名。

    返回:
        tuple: (solver, t_array, T_matrix)

    实现要求:
    1.  实例化 HeatDiffusionSolver (初始条件为 params['T0']，边界条件来自 params)。
    2.  调用 explicit_finite_difference 方法，并传入 h_coeff 和 T_env 参数。
    3.  如果 save_plot 为 True，则在 params['plot_times'] 指定的时间点绘制温度分布图。
    """
    # TODO: 实例化求解器
    # solver = HeatDiffusionSolver(L=params['L'], K=params['K'], C=params['C'], rho=params['rho'],
    #                            nx=params['nx'], total_time=params['total_time'], dt=params['dt'],
    #                            initial_condition_config=params['T0'],
    #                            boundary_conditions=((0, params['bc_left_val']), (0, params['bc_right_val'])))

    # TODO: 计算数值解，传入冷却参数
    # t_array, T_matrix = solver.explicit_finite_difference(h_coeff=params['h_coeff'], T_env=params['T_env'])

    # TODO: 如果 save_plot 为 True，则绘图 (类似 problem_1)
    # if save_plot:
    #     plt.figure(figsize=(10, 6))
    #     for t_plot in params['plot_times']:
    #         time_idx = np.argmin(np.abs(t_array - t_plot))
    #         plt.plot(solver.x, T_matrix[:, time_idx], label=f't = {t_array[time_idx]:.1f}s (目标 {t_plot}s)')
    #     plt.xlabel('位置 x (m)')
    #     plt.ylabel('温度 T (K)')
    #     plt.title(f'任务5：牛顿冷却定律 (h={params["h_coeff"]:.2e}, T_env={params["T_env"]}) (学生版)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(plot_filename)
    #     print(f"图像已保存到 {plot_filename}")
    #     plt.show()

    # return solver, t_array, T_matrix
    raise NotImplementedError(f"请在 {__file__} 中实现 solve_heat_diffusion_newton_cooling 函数")


if __name__ == '__main__':
    # --- 学生测试区 --- 
    # 在此区域，你可以创建参数字典并调用上面定义的 solve_... 函数来测试你的实现。

    # 任务1：基本热传导模拟 (学生测试)
    print("--- 学生测试：任务1：基本热传导模拟 ---")
    task1_params_student = {
        'L': 0.1, 'nx': 21, 'total_time': 100.0, 'dt': 0.1, # 根据项目说明调整参数
        'K': 237.0, 'C': 900.0, 'rho': 2700.0,
        'initial_condition_type': 'constant', 'T0_val': 100.0,
        'boundary_condition_left_type': 'dirichlet', 'bc_left_val': 0.0,
        'boundary_condition_right_type': 'dirichlet', 'bc_right_val': 0.0,
        'plot_times': [0, 10.0, 50.0, 100.0],
        'animation_filename_student': "student_task1_animation.gif"
    }
    try:
        print("尝试运行 solve_heat_diffusion_problem_1 (学生版)")
        # solver_task1_student, t_task1_student, T_matrix_task1_student = solve_heat_diffusion_problem_1(task1_params_student, save_plot=True, plot_filename_student="student_task1_results.png")
        # print("任务1 学生测试（尝试）运行完毕。如果实现了绘图，请检查 student_task1_results.png 和 student_task1_animation.gif")
        print("请取消注释并实现 HeatDiffusionSolver 类和 solve_heat_diffusion_problem_1 函数以进行测试。")
        pass
    except NotImplementedError as e:
        print(f"测试任务1失败: {e}")
    except Exception as e:
        print(f"测试任务1时发生其他错误: {e}")

    # 任务2：与解析解对比 (学生测试)
    print("\n--- 学生测试：任务2：与解析解对比 ---")
    task2_params_student = {
        'L': 0.1, 'nx': 51, 'total_time': 200.0, 'dt': 0.05, # 减小dt以获得更精确的数值解
        'K': 237.0, 'C': 900.0, 'rho': 2700.0,
        'initial_condition_type': 'sine_half_period', # 与解析解匹配的初始条件
        'T_max_sine': 100.0, # sin初始条件的最大温度
        'boundary_condition_left_type': 'dirichlet', 'bc_left_val': 0.0,
        'boundary_condition_right_type': 'dirichlet', 'bc_right_val': 0.0,
        'plot_time_analytical': 100.0, # 选择一个时间点进行对比
        'analytical_terms': 50 # 解析解级数项数
    }
    try:
        print("尝试运行 solve_heat_diffusion_analytical_comparison (学生版)")
        # solve_heat_diffusion_analytical_comparison(task2_params_student, save_plot=True, plot_filename_student="student_task2_analytical_comparison.png")
        # print("任务2 学生测试（尝试）运行完毕。如果实现了绘图，请检查 student_task2_analytical_comparison.png")
        print("请取消注释并实现相关函数以进行测试。")
        pass
    except NotImplementedError as e:
        print(f"测试任务2失败: {e}")
    except Exception as e:
        print(f"测试任务2时发生其他错误: {e}")

    # 任务3：稳定性分析 (学生测试) - 通常通过观察不同dt下的行为来完成，此处不直接编码测试，鼓励学生实验
    print("\n--- 学生测试：任务3：稳定性分析 ---")
    print("对于稳定性分析，请学生自行修改 task1_params_student 中的 'dt' 值进行实验。")
    print("例如，尝试一个较大的 dt (如 dt > dx^2 / (2*alpha)) 观察数值解是否发散。")

    # 任务4：不同初始条件 - 两根不同温度的铝棒接触 (学生测试)
    print("\n--- 学生测试：任务4：两根不同温度铝棒接触 ---")
    task4_params_student = {
        'L': 0.2, 'nx': 41, 'total_time': 500.0, 'dt': 0.2,
        'K': 237.0, 'C': 900.0, 'rho': 2700.0,
        'initial_condition_type': 'two_rods', 
        'T_left_rod': 100.0, 'T_right_rod': 0.0, 'contact_point_ratio': 0.5, # 接触点在中间
        'boundary_condition_left_type': 'dirichlet', 'bc_left_val': 100.0, # 假设左端保持100度
        'boundary_condition_right_type': 'dirichlet', 'bc_right_val': 0.0, # 假设右端保持0度
        'plot_times': [0, 50.0, 200.0, 500.0],
        'animation_filename_student': "student_task4_two_rods_animation.gif"
    }
    try:
        print("尝试运行 solve_heat_diffusion_problem_1 (学生版 - 任务4)")
        # solver_task4_student, t_task4_student, T_matrix_task4_student = solve_heat_diffusion_problem_1(task4_params_student, save_plot=True, plot_filename_student="student_task4_two_rods_results.png")
        # print("任务4 学生测试（尝试）运行完毕。如果实现了绘图，请检查 student_task4_two_rods_results.png 和 student_task4_two_rods_animation.gif")
        print("请取消注释并实现相关函数以进行测试。")
        pass
    except NotImplementedError as e:
        print(f"测试任务4失败: {e}")
    except Exception as e:
        print(f"测试任务4时发生其他错误: {e}")

    # 任务5：牛顿冷却定律 (学生测试)
    print("\n--- 学生测试：任务5：牛顿冷却定律 ---")
    task5_params_student = {
        'L': 0.1, 'nx': 21, 'total_time': 1000.0, 'dt': 1.0,
        'K': 237.0, 'C': 900.0, 'rho': 2700.0,
        'initial_condition_type': 'constant', 'T0_val': 100.0,
        'boundary_condition_left_type': 'newton', 'h_coeff_left': 10.0, 'T_env_left': 20.0,
        'boundary_condition_right_type': 'newton', 'h_coeff_right': 10.0, 'T_env_right': 20.0,
        'plot_times': [0, 100.0, 500.0, 1000.0],
        'animation_filename_student': "student_task5_newton_cooling_animation.gif"
    }
    try:
        print("尝试运行 solve_heat_diffusion_newton_cooling (学生版)")
        # solver_task5_student, t_task5_student, T_matrix_task5_student = solve_heat_diffusion_newton_cooling(task5_params_student, save_plot=True, plot_filename_student="student_task5_newton_cooling_results.png")
        # print("任务5 学生测试（尝试）运行完毕。如果实现了绘图，请检查 student_task5_newton_cooling_results.png 和 student_task5_newton_cooling_animation.gif")
        print("请取消注释并实现相关函数以进行测试。")
        pass
    except NotImplementedError as e:
        print(f"测试任务5失败: {e}")
    except Exception as e:
        print(f"测试任务5时发生其他错误: {e}")

    print("\n学生模板脚本的 __main__ 部分运行完毕。请实现所有标记为 TODO 的部分并取消注释测试代码进行验证。")

"""