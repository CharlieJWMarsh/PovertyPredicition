import pandas as pd
import numpy as np
import math
import Functions as f

# ds = np.array([[1, 2, 3, 4, 5], [6, 13, 6, 4, 2], [3, 4, 5, 6, 7], [9, 10, 11, 12, 13]])
ds = np.random.rand(5, 5)
print(ds, '\n')

normalised_ds = f.normalise_ds(ds)
print(normalised_ds, '\n')

ed_matrix = f.form_ed_matrix(normalised_ds)
print(ed_matrix, '\n')

similarity_matrix = f.similarity_scores(ed_matrix)
print(similarity_matrix, '\n')

threshold_matrix = f.threshold(2, similarity_matrix)
print(threshold_matrix, '\n')

laplacian_matrix = f.laplacian(threshold_matrix)
print(laplacian_matrix, '\n')

eigen_values, eigen_vectors = np.linalg.eig(laplacian_matrix)
print(eigen_values, '\n')
print(eigen_vectors, '\n')



