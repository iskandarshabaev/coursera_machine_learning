import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import minimize

def f(x):
    return np.sin(x / 5.) * np.exp(x / 10.) + 5. * np.exp(-x / 2.)


def find_min(x, f):
    return minimize(f, x, method='BFGS').fun


X = np.arange(1, 30, 0.01)
dots_X = np.array([i for i in range(2, 30, 2)])
dots_Y = np.array([find_min(i, f) for i in dots_X])

plt.plot(X, f(X))
plt.plot(2, find_min(2, f), marker='o')
plt.plot(27, find_min(27, f), marker='o')
plt.show()
