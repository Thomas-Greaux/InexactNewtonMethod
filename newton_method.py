'''
This is the main file
it executes the inexact newton method
on the function f(x) = sum from i = 1 to d {(x_i/i)^2}
  (with x_i the ith component of vector x)
with different d and doing only j steps for the conjugate gradient method
  (with j <= d)
'''


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import conjugate_gradient as cg
import function as func

eps = 0.01 #epsilon, the precision demanded

def norm(x):
    '''
    returns the norm of vector x
    '''
    return np.inner(x, x)

def alpha(k):
    '''
    returns the hyper parameter alpha, the learning rate
    '''
    return 0.005

def inexact_newton(w, i):
    '''
    execute the inexact newton method using the conjugate gradient
    returns the number of steps necessary to achieve the requiered precision
    '''
    k = 0
    while(func.f(w) >= eps):
        # This is the heart of the inexact newton
        # We follow the formula wk+1 = wk - alphak * gradient(wk) / hessian(wk)
        # But we use the conjugate gradient method that gives use " - gradient(wk) / hessian(wk) "
        w = w + alpha(k) * cg.conjugate_gradient(w, func.hessian(w), -func.gradient(w), i)
        k = k+1
    return k

'''
This block is the program,
is increases the dimension and number of steps for the conjugate gradient.
'''
max_d = 18
increment = 2
for d in range(2, max_d+1, increment):
    func.set_dimension(d)
    x0 = func.generate_x0()
    for i in range(2, d+1, increment):
        steps = inexact_newton(x0, i)
        print("#steps at dimension " + str(d) + " and conjugate " + str(i) + " = " + str(steps))
    print("")
