import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
p_class = data['Pclass'].value_counts()
print(round(p_class[1] * 100 / len(data), 2))