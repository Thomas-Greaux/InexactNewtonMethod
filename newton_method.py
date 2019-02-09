import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import conjugate_gradient as cg

eps = 0.001 #epsilon, the precision demanded

def f(w):
    return (w[0]-1)**2 + (w[1]-1)**2

def dfdx1(w):
    return 2*w[0] -2

def dfdx2(w):
    return 2*w[1] -2

def gradient(w):
    return np.array([dfdx1(w), dfdx2(w)])

def hessian(w):
    return np.array([[2, 0], [0, 2]])

def norm(x):
    return np.inner(x, x)

def alpha(k):
    return 0.01

def inexact_newton(w):
    k = 0
    while(f(w) >= eps):
        print("-- step " + str(k))
        print("w = ")
        print(w)
        print("f(w) = " + str(f(w)))
        # This is the heart of the inexact newton
        # We follow the formula wk+1 = wk - alphak * gradient(wk) / hessian(wk)
        # But we use the conjugate gradient method that gives use " - gradient(wk) / hessian(wk) "
        w = w + alpha(k) * cg.conjugate_gradient(w, hessian(w), -gradient(w)) 
        k = k+1
        #if k > 10:
        #    break

inexact_newton(np.array([-1, -3]))

w = [0, 0]
hw = [[0, 0], [0, 0]]

nb_points = 100
bound = 3

x1 = np.linspace(-bound, bound, nb_points)
x2 = np.linspace(-bound, bound, nb_points)

X1, X2 = np.meshgrid(x1, x2)
Y = f([X1, X2])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, Y, 50, cmap='binary')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()
