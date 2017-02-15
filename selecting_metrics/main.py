from numpy import linspace
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

data = load_boston()
X = scale(data.data)
y = data.target
kf = KFold(n_splits=5, shuffle=True, random_state=42)

maxValue = 0
maxP = 0

for p in linspace(1, 10, 200):
    regressor = KNeighborsRegressor(metric='minkowski', p=p,
                                    n_neighbors=5, weights='distance')
    cv = cross_val_score(regressor, X, y, cv=kf)
    maxV = max(cv)
    print('p: ', p, ' max: ', maxV)
    if (maxValue < maxV):
        maxValue = maxV
        maxP = p
print('------')
print(maxP, maxValue)
