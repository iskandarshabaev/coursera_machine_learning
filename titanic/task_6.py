import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
female_filtered_data = data.where(data['Sex'] == 'female').dropna(subset=['Sex'])
names = female_filtered_data['Name'].str.split(',', expand=True)

print(names[1].value_counts(sort=True))
