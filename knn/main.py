import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data = pandas.read_csv('wine.data')

data = data[['Class', 'Alcohol',
             'Malic_acid', 'Ash', 'Alcalinity_of_ash',
             'Magnesium', 'Total_phenols', 'Flavanoids',
             'Nonflavanoid_phenols', 'Proanthocyanins',
             'Color_intensity', 'Hue', 'OD280_OD315_of_diluted_wines',
             'Proline']] \
    .dropna(subset=['Class', 'Alcohol',
                    'Malic_acid', 'Ash', 'Alcalinity_of_ash',
                    'Magnesium', 'Total_phenols', 'Flavanoids',
                    'Nonflavanoid_phenols', 'Proanthocyanins',
                    'Color_intensity', 'Hue', 'OD280_OD315_of_diluted_wines',
                    'Proline'])
X = data[['Alcohol',
          'Malic_acid', 'Ash', 'Alcalinity_of_ash',
          'Magnesium', 'Total_phenols', 'Flavanoids',
          'Nonflavanoid_phenols', 'Proanthocyanins',
          'Color_intensity', 'Hue', 'OD280_OD315_of_diluted_wines',
          'Proline']].values
y = data['Class'].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)


def sumCross(X, y, k):
    clf = KNeighborsClassifier(n_neighbors=k)
    # clf.fit(X, y)
    # cv = ShuffleSplit(n_splits=5, random_state=42)
    cv = cross_val_score(clf, X, y, cv=kf)
    return cv.mean()


def co():
    maxVal = 0.0
    maxIndex = 0
    for k in range(1, 51):
        v = sumCross(X, y, k)
        print(k, v)
        if (v > maxVal):
            maxVal = v
            maxIndex = k
    return maxIndex, maxVal


print(co())
X = scale(X)
print('--------')
print(co())
