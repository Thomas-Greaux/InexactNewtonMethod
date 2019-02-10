import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import conjugate_gradient as cg
import function as func

eps = 0.001 #epsilon, the precision demanded

def norm(x):
    return np.inner(x, x)

def alpha(k):
    return 0.02

def inexact_newton(w, i, debug):
    '''
    execute the inexact newton method using the conjugate gradient
    '''
    k = 0
    while(func.f(w) >= eps):
        if(debug):
            print("w = " + str(w))

        # This is the heart of the inexact newton
        # We follow the formula wk+1 = wk - alphak * gradient(wk) / hessian(wk)
        # But we use the conjugate gradient method that gives use " - gradient(wk) / hessian(wk) "
        w = w + alpha(k) * cg.conjugate_gradient(w, func.hessian(w), -func.gradient(w), i)
        k = k+1
    return k

for d in range(2, 20):
    func.set_dimension(d)
    x0 = func.generate_x0()
    for i in range(3, d+1):
        steps = inexact_newton(x0, i, 0)
        print("#steps at dimension " + str(d) + " and conjugate " + str(i) + " = " + str(steps))
    print("")
