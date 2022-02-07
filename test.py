import pandas as pd
import numpy as np
import math
import Functions as f
import heapq


all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')

OAs = all_data['geography_code']
census_data = all_data.drop(['geography_code'], axis=1)
census_data = census_data.to_numpy()

print(census_data)

column_mean = np.sum(census_data, axis=0) / np.shape(census_data)[0]
print(column_mean)

column_sd = np.sqrt(np.sum((census_data - column_mean) ** 2, axis=0) / np.shape(census_data)[0])
normalised_census_data = (census_data - column_mean) / column_sd
faulty_columns = []
for i in range(0, np.shape(census_data)[1]):
    if np.isnan(np.sum(normalised_census_data[:, i])):
        faulty_columns.insert(0, i)
print("faulty columns: ", faulty_columns)
for i in range(0, len(faulty_columns)):
    normalised_census_data = np.delete(normalised_census_data, faulty_columns[i], 1)
# np.savetxt('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\safety_normalised_matrix.csv',
#            normalised_census_data, delimiter=',')
print("shape after normalise: ", np.shape(census_data))

