import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
survived = data['Survived'].value_counts()
print(round(survived[1] * 100 / len(data), 2))