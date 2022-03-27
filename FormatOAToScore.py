import pandas as pd
import numpy as np
import math
import heapq


def make_OA_score_data(eigenvalue):
    # Find the 2nd largest eigenvalue
    value = heapq.nlargest(1, heapq.nsmallest((eigenvalue + 1), eingenvalues))[0]
    value = float(value)

    # Find index of x largest eigenvalue
    index = np.where(eingenvalues == value)[0]

    # Get eigenvector of x largest eigenvalue
    component = eingenvectors[:, index]

    # Join OA to eigenvector
    ds = np.hstack((OAs, component))

    return ds


# Load eigenvector and eigenvalues from previous section
file1 = open('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\Bristol_eigenvalues.csv')
file2 = open('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\Bristol_eigenvectors.csv')
eingenvalues = np.loadtxt(file1, delimiter=",")
eingenvectors = np.loadtxt(file2, delimiter=",")

# Load OA's from input data
OAs = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\BristolData.csv')
OAs = OAs['geography_code']
OAs = OAs.to_numpy()
OAs = OAs.reshape(len(OAs), 1)

ds = make_OA_score_data(1)

print(ds)
print(np.shape(ds))

bristol_heatmap_data = pd.DataFrame(ds, columns=['OA', 'score'])

bristol_heatmap_data.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Bristolheatmap_alldata_2.csv', index=False)

