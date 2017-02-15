import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
male, female = data['Sex'].value_counts()
print(male, female)