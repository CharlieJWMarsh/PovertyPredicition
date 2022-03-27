import pandas as pd
import numpy as np
import Functions as f


all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\BristolData.csv')

OAs = all_data['geography_code']
census_data = all_data.drop(['geography_code'], axis=1)
census_data = census_data.to_numpy()

# print(np.shape(census_data))

# print(census_data[:, 146])
# census_data = census_data[:, 0:146]

eigen_values, eigen_vectors = f.diffusion_map(10, census_data)

# print(np.shape(eigen_values))
# print(np.shape(eigen_vectors))

eigen_values = np.asarray(eigen_values)
eigen_values = eigen_values.reshape(len(eigen_values), 1)

eigen_vectors = np.asarray(eigen_vectors)

# print(np.shape(eigen_values))
# print(np.shape(eigen_vectors))

np.savetxt('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\Bristol_eigenvalues_2.csv',
           eigen_values, delimiter=',')

np.savetxt('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\Bristol_eigenvectors_2.csv',
           eigen_vectors, delimiter=',')

# print(eigen_values, '\n')
# print(eigen_vectors, '\n')
