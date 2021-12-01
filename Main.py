import pandas as pd
import numpy as np
import math
import Functions as f

all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\most_data.csv')


OAs = all_data['geography_code']
census_data = all_data.drop(['geography_code'], axis=1)
census_data = census_data.to_numpy()

# census_data = census_data[0:8000, :]

eigen_values, eigen_vectors = f.diffusion_map(10, census_data)

eigen_values = np.asarray(eigen_values)
eigen_values = eigen_values.reshape(len(eigen_values), 1)

eigen_vectors = np.asarray(eigen_vectors)

np.savetxt('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\initial_eigen_values_2.csv',
           eigen_values, delimiter=',')

np.savetxt('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\initial_eigen_vectors_2.csv',
           eigen_vectors, delimiter=',')

print(eigen_values, '\n')
print(eigen_vectors, '\n')
