import math
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy


def diff(A, B):
    max = 0
    for i in range(1, N-1):
        for j in range(1, N-1):
            if abs(A[i][j] - B[i][j]) > max:
                max = abs(A[i][j] - B[i][j])
                
    return max


def start():
    
    U = [[0 for i in range(N)] for k in range(N)]
    for i in range(N):
        U[0][i] = round(20 * y_j(i), 3)
    for i in U:
        i[-1] = 20 
    for i in range(N):
        U[-1][i] = round(20 * y_j(i) ** 2, 3)
    for i in range(N):
        U[i][0] = round(50 * x_i(i) * (1 - x_i(i)), 3)
        
    return U

h = 0.2
l = 1
N = round(l / h) + 1
w = 1
e = 0.01

x_i = lambda i: i * h
y_j = lambda j: j * h

W = []
K = []
for w in np.arange(0.5, 2, 0.1):
    W.append(w)
    U = start()
    U_k = deepcopy(U)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            U[i][j] = U[i][j] + w * ( 0.25 * (U[i-1][j] +
             U[i+1][j] + U[i][j-1] + U[i][j+1]) - U[i][j] )
    k = 1
    while diff(U, U_k) > e:
        k += 1
        U_k = deepcopy(U)
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                U[i][j] = U[i][j] + w * ( 0.25 * (U[i-1][j] +
                 U[i+1][j] + U[i][j-1] + U[i][j+1]) - U[i][j] )

    K.append(k)

U = start()
U_k = deepcopy(U)
w = 1.3
for i in range(1, N - 1):
    for j in range(1, N - 1):
        U[i][j] = U[i][j] + w * ( 0.25 * (U[i-1][j] +
         U[i+1][j] + U[i][j-1] + U[i][j+1]) - U[i][j] )
k = 1
while diff(U, U_k) > e:
    k += 1
    U_k = deepcopy(U)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            U[i][j] = U[i][j] + w * ( 0.25 * (U[i-1][j] +
             U[i+1][j] + U[i][j-1] + U[i][j+1]) - U[i][j] )

plt.figure(figsize=(16, 9))
plt.plot(W, K, linewidth=5)
plt.plot([1.3], [10], 'ro')
plt.text(1.25, 12, '$\omega=1.3$', fontsize=20)
plt.title('Title', fontsize=20)
plt.xlabel('$\omega$', fontsize=20)
plt.ylabel('$k$', fontsize=20)
plt.grid()

fig = plt.figure(figsize=(16, 9))
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, l + h/10, h)
X, Y = np.meshgrid(X, X)
U = np.array(U)

# Plot the surface.
ax.plot_surface(X, Y, np.array((U)), cmap='cool')
ax.plot_wireframe(X, Y, np.array((U)), color='black')

ax.set_xlabel("y", fontsize=20)
ax.set_ylabel("x", fontsize=20)
ax.set_zlabel("U", fontsize=20)
ax.view_init(30, 230)

fig.suptitle('title', fontsize=20)