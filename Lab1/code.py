import math
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, 
    FormatStrFormatter
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


h = 0.1
l = 0.6
N = round(l / h) + 1

t_0 = 0.01
dt = 0.001
K = round(t_0 / dt) + 1

x_i = lambda i: i * h
t_j = lambda j: j * dt

T = [[None for i in range(N)] for k in range(K)]

# set u(x;0) = 1 - lg(x + 0.4)
f = lambda x: 1 - math.log10(x + 0.4)
for i in range(len(T[0])):
    T[0][i] = f(x_i(i))

# set u(0;t) = 1.4
for i in range(len(T)):
    T[i][0] = 1.4

# set u(0.6;t) = t + 1
for i in range(len(T)):
    T[i][-1] = t_j(i) + 1

for k in range(K-1):
    for i in range(1, N-1):
        T[k+1][i] = (T[k][i-1] - 2 * T[k][i] +
             T[k][i+1]) * dt / (h ** 2) + T[k][i]


fig = plt.figure(figsize=(16, 9))
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, l + h/10, h)
t = np.arange(0, t_0 + dt/10, dt)
X, t = np.meshgrid(X, t)
T = np.array(T)

# Plot the surface.
ax.plot_surface(X, t, np.array((T)),
     cmap='cool')
ax.plot_wireframe(X, t, np.array((T)),
     color='black')

ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.set_zlabel("T", fontsize=20)
ax.view_init(30, 40)

fig.suptitle('T(t, x)', fontsize=20)


fig = plt.figure(figsize=(16, 9))
plt.grid()
plt.title('T(t, x)', fontsize=20)
plt.xlabel('x', fontsize=20)
plt.ylabel('T', fontsize=20)

x = np.arange(0, l + h/10, h)

y1 = np.array(T[0])
plt.plot(x, y1, linewidth=2, 
    label='t = %.3f' % t_j(0), linestyle='-', marker='o')

y2 = np.array(T[3])
plt.plot(x, y2, linewidth=2, 
    label='t = %.3f' % t_j(3), linestyle='-', marker='o')

y3 = np.array(T[7])
plt.plot(x, y3, linewidth=2, 
    label='t = %.3f' % t_j(7), linestyle='-', marker='o')

y4 = np.array(T[10])
plt.plot(x, y4, linewidth=2, 
    label='t = %.3f' % t_j(10), linestyle='-', marker='o')

plt.legend(fontsize=16)