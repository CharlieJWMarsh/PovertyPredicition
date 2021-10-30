import pandas as pd
import numpy as np
import math
import Functions as f

ds = np.array([[1, 2, 3, 4, 5], [6, 13, 6, 4, 2], [3, 4, 5, 6, 7], [9, 10, 11, 12, 13]])
print(ds, '\n')

normalised_ds = f.normalise_ds(ds)
print(normalised_ds, '\n')

ed_matrix = f.form_ed_matrix(normalised_ds)
print(ed_matrix, '\n')

similarity_matrix = f.similarity_scores(ed_matrix)
print(similarity_matrix, '\n')
