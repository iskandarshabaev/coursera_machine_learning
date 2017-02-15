import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import differential_evolution
from scipy.optimize import minimize


def f(x):
    return np.sin(x / 5.) * np.exp(x / 10.) + 5. * np.exp(-x / 2.)


def h(x):
    return np.int(f(x))


def find_minBFGS(x, f):
    return minimize(f, x, method='BFGS')


def find_min(x, f):
    return differential_evolution(f, x)


differ = find_min([(0, 30)], h)
bfgs = find_minBFGS(30, h)

X = np.arange(1, 30, 0.01)
h_X = np.array([h(i) for i in X])

plt.plot(X, h_X)
plt.plot(bfgs.x, bfgs.fun, marker='o')
plt.plot(differ.x, differ.fun, marker='o')
plt.show()
