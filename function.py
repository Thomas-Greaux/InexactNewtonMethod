import numpy as np

dimension = 3

def set_dimension(d):
  global dimension
  dimension = d

def generate_x0():
  x0 = np.array([])
  for i in range(dimension):
    x0 = np.append(x0, 10)
  return x0

def f(w):
  res = 0
  for i in range(dimension):
    res += w[i]**2 / float((i+1)**2)
  return res

def gradient(w):
  grad = np.array([])
  for i in range(dimension):
    value = float(2*w[i]/float((i+1)**2))
    grad = np.append(grad, value)
  return grad

def help_hessian(j):
  res = np.array([])
  for i in range(dimension):
    value = 0.0
    if i==j:
      value = 2.0*(1.0/((i+1.0)**2))
    res = np.append(res, value)
  return res

def hessian(w):
  hessian = np.array([help_hessian(0)])
  for i in range(1, dimension):
    hessian = np.append(hessian, np.array([help_hessian(i)]), axis=0)
  return hessian
