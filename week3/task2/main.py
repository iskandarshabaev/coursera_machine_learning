import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import differential_evolution


def f(x):
    return np.sin(x / 5.) * np.exp(x / 10.) + 5. * np.exp(-x / 2.)


def find_min(x, f):
    return differential_evolution(f, x).fun


X = np.arange(1, 30, 0.01)
dots_X = np.array([i for i in range(2, 30, 2)])

plt.plot(X, f(X))
plt.plot(4, find_min([(0, 10)], f), marker='o')
plt.show()
