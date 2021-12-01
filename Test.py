import pandas as pd
import numpy as np
import math
import Functions as f

# all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\all_data.csv')

# print(all_data.head())
# print(all_data.shape)
#
# OAs = all_data['geography_code']
#
# print(OAs.head())
# print(OAs.shape)
#
# census_data = all_data.drop(['geography_code'], axis=1)
#
# print(census_data.head())
# print(census_data.shape)
#
# census_data = census_data.to_numpy()
# print(census_data)

# ds = np.array([[1, 2, 3, 4, 5], [6, 13, 6, 4, 2], [3, 4, 5, 6, 7], [9, 10, 11, 12, 13]])
ds = np.random.rand(1000, 100)
print(ds, '\n')
print("ds", '\n', ds, '\n')

eigen_values, eigen_vectors = f.diffusion_map(ds)

eigen_values = np.asarray(eigen_values)
eigen_values = eigen_values.reshape(len(eigen_values), 1)

eigen_vectors = np.asarray(eigen_vectors)

print(eigen_values)
print(eigen_values.shape)

print(eigen_vectors)
print(eigen_vectors.shape)


