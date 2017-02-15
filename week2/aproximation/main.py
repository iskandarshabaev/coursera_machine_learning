import numpy as np
import scipy.linalg as la
from matplotlib import pylab as plt


def f(x):
    return np.sin(x / 5.) * np.exp(x / 10.) + 5. * np.exp(-x / 2.)


def ff(x, k):
    w_0 = k[0]
    array = [k[i] * x ** float(i) for i in range(1, len(k))]
    return sum(array) + w_0


def solve(x):
    n = len(x)
    A = np.array([[x[i] ** a for a in range(0, n)] for i in range(n)])
    b = np.array([f(x[i]) for i in range(n)])
    return la.solve(A, b)


k2 = solve([1., 15.])
k3 = solve([1., 8., 15.])
k4 = solve([1., 4., 10., 15.])
X = np.arange(1, 15, 0.01)

plt.plot(X, f(X), X, ff(X, k4))
plt.show()
