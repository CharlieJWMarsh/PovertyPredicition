import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px


all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')
poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")

# Takes data from csv without OAs
census_data = all_data.drop(['geography_code'], axis=1)
census_data = census_data.to_numpy()

# print(census_data)
print(np.shape(census_data))

pca = PCA(n_components=10, random_state=32456)

processed_data = pca.fit(census_data.T)

eigenvalues = processed_data.explained_variance_
eigenvectors = processed_data.components_

print(eigenvalues)
print(np.shape(eigenvalues))
# print(eigenvectors)
print(np.shape(eigenvectors))

# print(eigenvectors[:, 0])
# print(eigenvectors[0, :])

df = eigenvectors.T
df = pd.DataFrame(df, columns=['Eigenvector 1', 'Eigenvector 2', 'Eigenvector 3', 'Eigenvector 4', 'Eigenvector 5',
                               'Eigenvector 6', 'Eigenvector 7', 'Eigenvector 8', 'Eigenvector 9', 'Eigenvector 10'])

df = pd.concat([df, poverty_score_data["score"]], axis=1)
df = df.rename(columns={"score": "Deprivation score"})


print(df.head())
print(np.shape(df))

# fig = px.scatter(df, x="Eigenvector 6", y="Eigenvector 7", color="Deprivation score")
# fig.show()

fig1 = px.scatter_3d(df, x="Eigenvector 5", y="Eigenvector 6", z="Eigenvector 7", color="Deprivation score")
fig1.update_traces(marker_size=3)

fig1.show()
