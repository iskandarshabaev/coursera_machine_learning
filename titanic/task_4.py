import pandas
import numpy as np

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
filtered_age_data = [x for x in data['Age'].values if not np.isnan(x)]
average = np.mean(filtered_age_data)
median = np.median(filtered_age_data)
print(average, median)
