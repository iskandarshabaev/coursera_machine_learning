import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col='PassengerId').dropna(subset=['Pclass', 'Fare', 'Age', 'Sex'])
data['Sex'] = (data['Sex'] == 'male').astype(int)
X = data[['Pclass', 'Fare', 'Age', 'Sex']]
y = data[['Survived']]
print(X.values)
print(y.values)
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X.values, y.values)
importances = clf.feature_importances_
print(importances)
