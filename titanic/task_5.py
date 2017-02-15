import pandas
import numpy as np
import math

data = pandas.read_csv('titanic.csv', index_col='PassengerId')


def pirson_k(X, Y):
    x_sum = np.sum(X)
    y_sum = np.sum(Y)
    x_mid = x_sum / len(X)
    y_mid = y_sum / len(Y)
    X_dev_sum = np.sum([np.power(x_mid - i, 2) for i in X])
    Y_dev_sum = np.sum([np.power(y_mid - i, 2) for i in Y])
    dev_sum = 0
    for i in range(0, len(X)):
        dev_sum += (x_mid - X[i]) * (y_mid - Y[i])
    return dev_sum / math.sqrt(X_dev_sum * Y_dev_sum)


sib_sp = data['SibSp'].values
parch = data['Parch'].values

print(pirson_k(sib_sp, parch))
