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

'''
This block is the parsing of arguments
'''

import argparse

parser = argparse.ArgumentParser(description="Execute the inexact newton method using the conjugate gradient method")
parser.add_argument("-m", "--maxd", type=int, metavar="", default=18, help="The maximum dimension of the function")
parser.add_argument("-i", "--increment", type=int, metavar="", default=2, help="The increment of the number of steps in the conjugate gradient method")
parser.add_argument("-e", "--epsilon", type=float, metavar="", default=0.01, help="Epsilon, the precision demanded")
parser.add_argument("-a", "--alpha", type=float, metavar="", default=0.005, help="Alpha, the learning rate")

args = parser.parse_args()

max_d = args.maxd
increment = args.increment
epsilon = args.epsilon
alpha = args.alpha

def norm(x):
    '''
    returns the norm of vector x
    '''
    return np.inner(x, x)

def inexact_newton(w, i):
    '''
    execute the inexact newton method using the conjugate gradient
    returns the number of steps necessary to achieve the requiered precision
    '''
    k = 0
    while(func.f(w) >= epsilon):
        # This is the heart of the inexact newton
        # We follow the formula wk+1 = wk - alphak * gradient(wk) / hessian(wk)
        # But we use the conjugate gradient method that gives use " - gradient(wk) / hessian(wk) "
        w = w + alpha * cg.conjugate_gradient(w, func.hessian(w), -func.gradient(w), i)
        k = k+1
    return k

'''
This block is welcoming the user
'''
print("Executing with arguments : ")
print("max d = " + str(max_d))
print("increment = " + str(increment))
print("epsilon = " + str(epsilon))
print("alpha = " + str(alpha))
print("")

'''
This block is the program,
is increases the dimension and number of steps for the conjugate gradient.
'''
for d in range(2, max_d+1, increment):
    func.set_dimension(d)
    x0 = func.generate_x0()
    for i in range(2, d+1, increment):
        steps = inexact_newton(x0, i)
        print("#steps at dimension " + str(d) + " and conjugate " + str(i) + " = " + str(steps))
    print("")
