import pandas as pd
import numpy as np
import math
import Functions as f
import heapq


np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

# ds = np.random.rand(1, 1000)
# ds = ds[0, :]
# ds = ds.tolist()
# print(type(ds))
#
# largest_ten = heapq.nlargest(10, ds)
# tenth_largest = heapq.nsmallest(1, largest_ten)
#
# print(largest_ten)
# print(tenth_largest)

# all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\all_data.csv')
all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\most_data.csv')


OAs = all_data['geography_code']
census_data = all_data.drop(['geography_code'], axis=1)
census_data = census_data.to_numpy()

array_sum = np.sum(census_data)
array_has_nan = np.isnan(array_sum)

print(array_has_nan)

print(census_data.shape)

census_data = census_data[0:10, 0:10]

print(census_data)

# ds = np.random.randint(100, size=(10, 6))
# print(ds)

eigen_values, eigen_vectors = f.diffusion_map(10, census_data)

eigen_values = np.asarray(eigen_values)
eigen_values = eigen_values.reshape(len(eigen_values), 1)

eigen_vectors = np.asarray(eigen_vectors)

print(eigen_values)
print(eigen_vectors)

