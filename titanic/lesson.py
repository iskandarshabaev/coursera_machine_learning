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
    return dev_sum/math.sqrt(X_dev_sum * Y_dev_sum)


male, female = data['Sex'].value_counts()
print(male, ' ', female)
survived = data['Survived'].value_counts()
print(round(survived[1] * 100 / len(data), 2))
p_class = data['Pclass'].value_counts()
print(round(p_class[1] * 100 / len(data), 2))
filtered_age_data = [x for x in data['Age'].values if not np.isnan(x)]
average = np.mean(filtered_age_data)
median = np.median(filtered_age_data)
print(average, ' ', median)
print (pirson_k([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]))

print(data['Survived'].value_counts())
