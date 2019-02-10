import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import conjugate_gradient as cg

eps = 0.001 #epsilon, the precision demanded

a = 2.0
b = 5.0
c = 20.0

def f(w):
    return (w[0]/a)**2 + (w[1]/b)**2 + (w[2]/c)**2

def gradient(w):
    return np.array([2*w[0]/(a**2), 2*w[1]/(b**2), 2*w[2]/(c**2)])

def hessian(w):
    return np.array([
        [2/(a**2), 0, 0],
        [0, 2/(b**2), 0],
        [0, 0, 2/(c**2)]])

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
        print("## f(w) = " + str(f(w)))
        # This is the heart of the inexact newton
        # We follow the formula wk+1 = wk - alphak * gradient(wk) / hessian(wk)
        # But we use the conjugate gradient method that gives use " - gradient(wk) / hessian(wk) "
        tmp = f(w)
        res_cg = cg.conjugate_gradient(w, hessian(w), -gradient(w)) 
        w = w + alpha(k) * res_cg
        print(res_cg)
        tmp -= f(w)

        if tmp < -10:
            break
        k = k+1

x0 = np.array([-4, 2, 10])
inexact_newton(x0)
