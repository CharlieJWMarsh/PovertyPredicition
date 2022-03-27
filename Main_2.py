import pandas as pd
import numpy as np
import Functions_2 as f
import sys

# Takes argument from command line
all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')

# Takes OAs
OAs = all_data['geography_code']
OAs = OAs.to_numpy()
OAs = OAs.reshape(len(OAs), 1)

# Takes data from csv without OAs
census_data = all_data.drop(['geography_code'], axis=1)
census_data = census_data.to_numpy()

# Performs diffusion map method
eigen_values, eigen_vectors = f.diffusion_map(10, census_data)

# Changes form of eigenvalues
eigen_values = np.asarray(eigen_values)
eigen_values = eigen_values.reshape(len(eigen_values), 1)

# Changes form of eigenvectors
eigen_vectors = np.asarray(eigen_vectors)

# Gets the 10 most important eigenvalues eigenvectors
ds_1 = f.make_OA_score_data(1, eigen_values, eigen_vectors, OAs)
ds_2 = f.make_OA_score_data(2, eigen_values, eigen_vectors, OAs)
ds_3 = f.make_OA_score_data(3, eigen_values, eigen_vectors, OAs)
ds_4 = f.make_OA_score_data(4, eigen_values, eigen_vectors, OAs)
ds_5 = f.make_OA_score_data(5, eigen_values, eigen_vectors, OAs)
ds_6 = f.make_OA_score_data(6, eigen_values, eigen_vectors, OAs)
ds_7 = f.make_OA_score_data(7, eigen_values, eigen_vectors, OAs)
ds_8 = f.make_OA_score_data(8, eigen_values, eigen_vectors, OAs)
ds_9 = f.make_OA_score_data(9, eigen_values, eigen_vectors, OAs)
ds_10 = f.make_OA_score_data(10, eigen_values, eigen_vectors, OAs)

# Formats the 10 most important eigenvalues eigenvectors
bristol_heatmap_data_1 = pd.DataFrame(ds_1, columns=['OA', 'score'])
bristol_heatmap_data_2 = pd.DataFrame(ds_2, columns=['OA', 'score'])
bristol_heatmap_data_3 = pd.DataFrame(ds_3, columns=['OA', 'score'])
bristol_heatmap_data_4 = pd.DataFrame(ds_4, columns=['OA', 'score'])
bristol_heatmap_data_5 = pd.DataFrame(ds_5, columns=['OA', 'score'])
bristol_heatmap_data_6 = pd.DataFrame(ds_6, columns=['OA', 'score'])
bristol_heatmap_data_7 = pd.DataFrame(ds_7, columns=['OA', 'score'])
bristol_heatmap_data_8 = pd.DataFrame(ds_8, columns=['OA', 'score'])
bristol_heatmap_data_9 = pd.DataFrame(ds_9, columns=['OA', 'score'])
bristol_heatmap_data_10 = pd.DataFrame(ds_10, columns=['OA', 'score'])

bristol_heatmap_data_1.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_1.csv', index=False)
bristol_heatmap_data_2.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv', index=False)
bristol_heatmap_data_3.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_3.csv', index=False)
bristol_heatmap_data_4.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_4.csv', index=False)
bristol_heatmap_data_5.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_5.csv', index=False)
bristol_heatmap_data_6.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_6.csv', index=False)
bristol_heatmap_data_7.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_7.csv', index=False)
bristol_heatmap_data_8.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_8.csv', index=False)
bristol_heatmap_data_9.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_9.csv', index=False)
bristol_heatmap_data_10.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_10.csv', index=False)
