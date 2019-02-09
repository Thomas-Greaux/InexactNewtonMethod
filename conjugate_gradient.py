import numpy as np

A = np.array([[],[]])
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
    the positive definite matrix A
    '''
    return np.dot(np.dot(X.T, A), Y)

def A_orthogonal_set(grad_xt, pt):
    '''
    This function returns the next vector p
    A-orthogonal to all the previous p
    based on equation 2.22 of Bubeck
    '''
    return grad_xt - A_inner_product(grad_xt, pt) * pt / A_inner_product(pt, pt)

def conjugate_gradient(x, A_ext, b_ext):
    '''
    This function returns the exact minimum of
    f(x) = 1/2 * x.T A x - b.T x,
    which is equivalent of solving Ax = b, 
    with the conjugate gradient method, 
    starting from x
    with A = nabla^2 F and b = - nabla F
    '''
    global A
    global b
    A = A_ext
    b = b_ext

    xt = x
    print(xt)
    p = [gradient(x)]
    d = xt.size
    for i in range(1, d):
        grad_xt = gradient(xt)
        xt = xt - np.inner(grad_xt, p[i-1]) * p[i-1] / A_inner_product(p[i-1], p[i-1])
        p.append(A_orthogonal_set(grad_xt, p[i-1]))
    return xt

#A = np.array(
#    [[2, 0],
#     [0, 2]]
#)

#b = np.array([4, -2])

if __name__=="__main__":
    x = np.array([-14, -2])
    x_star = conjugate_gradient(x)
    print("x = ", x)
    print("x* = ", x_star)
