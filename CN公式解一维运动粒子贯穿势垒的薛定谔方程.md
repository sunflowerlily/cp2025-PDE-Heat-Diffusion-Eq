一维运动粒子贯穿势垒的薛定谔方程为
$$i\hbar \frac{\partial\varphi}{\partial t} = -\frac{\hbar}{2m}\nabla^2\varphi+V\varphi=H\varphi$$
算符$H=\big(-\frac{\hbar}{2m}\frac{\partial^2}{\partial x^2}+V\big)$
取$\hbar=1, 2m=1$, 化简方程得
$$\frac{\partial\varphi}{\partial t} = -i\big(-\frac{\partial^2}{\partial x^2}+V\big)\varphi=-iH\varphi$$

利用平均公式求解，得到
$$\varphi_{i,j+1}=\bigg(\frac{1-i\frac{1}{2}H\Delta t}{1+i\frac{1}{2}H\Delta t}\bigg)\varphi_{i,j}=\bigg(\frac{2}{1+i\frac{1}{2}H\Delta t}-1\bigg)\varphi_{i,j} = \frac{2\varphi_{i,j}}{1+i\frac{1}{2}H\Delta t}-\varphi_{i,j}$$

令$$\chi=\frac{2\varphi_{i,j}}{1+i\frac{1}{2}H\Delta t}$$
上式可化为
$$\varphi_{i,j+1} = \chi-\varphi_{i,j}$$
且有
$$\big(1+\frac{i}{2}H\Delta t\big)\chi=2\varphi_{i,j}$$

利用差分格式，有
$$H\chi=\big(-\frac{\partial^2}{\partial x^2}+V\big)\chi =\frac{1}{h^2}( \chi_{i+1}-2\chi_{i}+\chi_{i}） + V\chi_{i}$$
其中$h$为$x$方向步长

则$$(1+i\frac{1}{2}H\Delta t)\chi=2\varphi_{i,j}$$
可以显式地写为
$$-\frac{i\Delta t}{2h^2}\chi_{i+1}+\bigg(1+\frac{i\Delta t}{h^2} +\frac{i\Delta t}{2}V\bigg)\chi_{i}-\frac{i\Delta t}{2h^2}\chi_{i-1}=2\varphi_{i,j}$$
上式除以$-i\Delta t/2h^2$, 化为
$$\chi_{i+1}+\bigg(-2+\frac{2ih^2}{\Delta t}-h^2V\bigg)\chi_i+\chi_{i-1} = \frac{4ih^2}{\Delta t}\varphi_{i,j}$$

构造3对角矩阵
$$\begin{pmatrix}
-2+\frac{2ih^2}{\Delta t}-h^2V&1&\ldots&0\\
1&-2+\frac{2ih^2}{\Delta t}-h^2V&\ldots&0\\
\vdots &\vdots & &\vdots\\
0&0&\ldots&-2+\frac{2ih^2}{\Delta t}-h^2V\\
\end{pmatrix}$$


```python
import numpy as np, matplotlib.pyplot as plt

def wavefun(x,x0=40,k0=0.5,d=10):
    '''高斯波包函数'''
    return np.exp(k0*1j*x)*np.exp(-(x-x0)**2*np.log10(2)/d**2)

Nx = 220 #空间坐标的栅格化点数
x = np.arange(Nx)
V = np.zeros(Nx)  #势函数
BW = 3 #势垒宽度
BH = 1.0 #势垒高度
V[Nx//2:Nx//2+BW] = BH   #构建势垒
Nt = 300   #时间栅格化点数

#取dt, h = 1, 则左侧3对角矩阵为
A = np.diag(-2+2j-V) + np.diag(np.ones(Nx-1),1) + np.diag(np.ones(Nx-1),-1) 

C = np.zeros((x.size, Nt), complex)   #chi 矩阵
B = np.zeros((x.size, Nt), complex)   #波函数矩阵

B0 = wavefun(x)  #初始波包
B[:,0] = B0.T

for t in range(Nt-1):
    C[:,t+1] = 4j*np.linalg.solve(A,B[:,t])
    B[:,t+1] = C[:,t+1] - B[:,t]

#以下动画实现
from matplotlib import animation
fig = plt.figure()
plt.axis([0, Nx, 0, BH*1.1])
myline, = plt.plot([],[],'r',lw=2)
myline1, = plt.plot(x, V, 'k', lw=2)
def animate(i):
    myline.set_data(x, np.abs(B[:,i]))
    myline1.set_data(x, V)
    return myline,myline1

anim=animation.FuncAnimation(fig,animate,frames=Nt, interval=20)
plt.show()
```
