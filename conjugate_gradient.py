'''
This file's purpose is to do
the conjugate gradient method
'''

import numpy as np

A = np.array([[]])
b = np.array([])

def gradient(x):
    '''
    Gradient of the f(x) = 1/2 * x.T A x - b.T x
    i.e. f'(x) = Ax -b
    '''
    return np.dot(A, x) - b

def A_inner_product(X, Y):
    '''
    This function returns the inner produxt defined by
    the positive definite matrix A, 
    i.e. <X, Y>_A = X.T * A * Y
    '''
    return np.dot(np.dot(X.T, A), Y)

def A_orthogonal_set(grad_xt, pt):
    '''
    This function returns the next vector p
    A-orthogonal to all the previous p
    based on equation 2.22 of Bubeck (section 2.4)
    '''
    return grad_xt - A_inner_product(grad_xt, pt) * pt / A_inner_product(pt, pt)

def conjugate_gradient(x, A_ext, b_ext, i):
    '''
    This function returns the minimum of
    f(x) = 1/2 * x.T A x - b.T x,
    which is equivalent of solving Ax = b, 
    with the conjugate gradient method, 
    starting at x, and doing it for i steps.
    For i = d, the dimension of x, it returns the exact minimum.
    '''
    global A
    global b
    A = A_ext
    b = b_ext

    xt = x
    p = gradient(x)
    for k in range(1, i):
        xt = xt - np.inner(gradient(xt), p) * p / A_inner_product(p, p)
        p = A_orthogonal_set(gradient(xt), p)
    return xt
