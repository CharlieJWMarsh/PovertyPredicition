import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
from sklearn import svm
import plotly.graph_objects as go


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
Perform PCA and get data
"""

# all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')
# poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
#
# # Takes data from csv without OAs
# census_data = all_data.drop(['geography_code'], axis=1)
# census_data = census_data.to_numpy()
#
# # print(census_data)
# print(np.shape(census_data))
#
# pca = PCA(n_components=10, random_state=32456)
#
# processed_data = pca.fit(census_data.T)
#
# eigenvalues = processed_data.explained_variance_
# eigenvectors = processed_data.components_
#
# print(eigenvalues)
# print(np.shape(eigenvalues))
# # print(eigenvectors)
# print(np.shape(eigenvectors))
#
# # print(eigenvectors[:, 0])
# # print(eigenvectors[0, :])
#
# df = eigenvectors.T
# df = pd.DataFrame(df, columns=['Eigenvector 1', 'Eigenvector 2', 'Eigenvector 3', 'Eigenvector 4', 'Eigenvector 5',
#                                'Eigenvector 6', 'Eigenvector 7', 'Eigenvector 8', 'Eigenvector 9', 'Eigenvector 10'])
#
# df = pd.concat([df, poverty_score_data["score"]], axis=1)
# df = df.rename(columns={"score": "Deprivation score"})
#
# OAs = all_data['geography_code']
# df['OA'] = OAs
# print(OAs)
#
#
# print(df.head())
# print(np.shape(df))
#
# df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_eigenvectors.csv', index=False)

"""
Linear plots
"""

# pca_evs = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_eigenvectors.csv')
# print(pca_evs.head())
#
# fig1 = px.scatter(pca_evs, hover_name="OA", x="Deprivation score", y="Eigenvector 1", trendline="ols")
# fig1.update_layout(font=dict(size=32))
# fig2 = px.scatter(pca_evs, hover_name="OA", x="Deprivation score", y="Eigenvector 2", trendline="ols")
# fig2.update_layout(font=dict(size=32))
# fig3 = px.scatter(pca_evs, hover_name="OA", x="Deprivation score", y="Eigenvector 3", trendline="ols")
# fig3.update_layout(font=dict(size=32))
# fig4 = px.scatter(pca_evs, hover_name="OA", x="Deprivation score", y="Eigenvector 4", trendline="ols")
# fig4.update_layout(font=dict(size=32))
# fig5 = px.scatter(pca_evs, hover_name="OA", x="Deprivation score", y="Eigenvector 5", trendline="ols")
# fig5.update_layout(font=dict(size=32))
# fig6 = px.scatter(pca_evs, hover_name="OA", x="Deprivation score", y="Eigenvector 6", trendline="ols")
# fig6.update_layout(font=dict(size=32))
# fig7 = px.scatter(pca_evs, hover_name="OA", x="Deprivation score", y="Eigenvector 7", trendline="ols")
# fig7.update_layout(font=dict(size=32))
# fig8 = px.scatter(pca_evs, hover_name="OA", x="Deprivation score", y="Eigenvector 8", trendline="ols")
# fig8.update_layout(font=dict(size=32))
# fig9 = px.scatter(pca_evs, hover_name="OA", x="Deprivation score", y="Eigenvector 9", trendline="ols")
# fig9.update_layout(font=dict(size=32))
# fig10 = px.scatter(pca_evs, hover_name="OA", x="Deprivation score", y="Eigenvector 10", trendline="ols")
# fig10.update_layout(font=dict(size=32))
#
#
# fig1.show()
# fig2.show()
# fig3.show()
# fig4.show()
# fig5.show()
# fig6.show()
# fig7.show()
# fig8.show()
# fig9.show()
# fig10.show()

"""
3D plots
"""

# pca_evs = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_eigenvectors.csv')
# print(pca_evs.head())
#
# fig1 = px.scatter_3d(pca_evs, x="Eigenvector 5", y="Eigenvector 6", z="Eigenvector 7", color="Deprivation score")
# fig1.update_traces(marker_size=3)
# fig1.update_layout(font=dict(size=18))
# fig2 = px.scatter_3d(pca_evs, x="Eigenvector 5", y="Eigenvector 6", z="Eigenvector 9", color="Deprivation score")
# fig2.update_traces(marker_size=3)
# fig2.update_layout(font=dict(size=18))
# fig3 = px.scatter_3d(pca_evs, x="Eigenvector 5", y="Eigenvector 7", z="Eigenvector 9", color="Deprivation score")
# fig3.update_traces(marker_size=3)
# fig3.update_layout(font=dict(size=18))
# fig4 = px.scatter_3d(pca_evs, x="Eigenvector 6", y="Eigenvector 7", z="Eigenvector 9", color="Deprivation score")
# fig4.update_traces(marker_size=3)
# fig4.update_layout(font=dict(size=18))
#
# fig1.show()
# fig2.show()
# fig3.show()
# fig4.show()

"""
Get AUC score and plot
"""

# pca_evs = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_eigenvectors.csv')
# print(pca_evs.head())
#
#
# ########################################################################################################
# #
# #
# #       Classify OAs for a for a certain percentage in poverty
# #
# #
# ########################################################################################################
#
# TPR_list = []
# FPR_list = []
#
# for i in range(1, 99):
#
#     print(i)
#
#     number_in_poverty = round((len(pca_evs) / 100) * i)
#
#     sorted_pca_evs = pca_evs.sort_values("Deprivation score", ascending=False)
#
#     classification = []
#     count = 0
#     for j in range(0, len(sorted_pca_evs)):
#         if count < number_in_poverty:
#             classification.append(1)
#         else:
#             classification.append(0)
#         count += 1
#
#     sorted_pca_evs['classification'] = classification
#
#     ########################################################################################################
#     #
#     #
#     #       plot data to show the classification
#     #
#     #
#     ########################################################################################################
#
#     # fig1 = px.scatter_3d(df, x="Eigenvector 5", y="Eigenvector 6", z="Eigenvector 9", color="classification", color_continuous_scale="Jet")
#     # fig1.update_traces(marker_size=3)
#     # fig1.show()
#
#     ########################################################################################################
#     #
#     #
#     #
#     #
#     #
#     ########################################################################################################
#
#     x = sorted_pca_evs[["Eigenvector 5", "Eigenvector 6", "Eigenvector 9"]].to_numpy()
#     y = sorted_pca_evs["classification"].to_numpy()
#
#     clf = svm.SVC(kernel='linear', C=100000, random_state=32456)
#     clf.fit(x, y)
#
#     predictions = clf.predict(x)
#
#     true_positive = 0
#     false_positive = 0
#     true_negative = 0
#     false_negative = 0
#
#     for i in range(0, len(predictions)):
#         if predictions[i] == 0 and classification[i] == 0:
#             true_negative += 1
#         elif predictions[i] == 1 and classification[i] == 1:
#             true_positive += 1
#         elif predictions[i] == 1 and classification[i] == 0:
#             false_positive += 1
#         elif predictions[i] == 0 and classification[i] == 1:
#             false_negative += 1
#
#     # print("true positives: ", true_positive)
#     # print("false positives: ", false_positive)
#     # print("true negatives: ", true_negative)
#     # print("false negatives: ", false_negative)
#
#     if (true_positive + false_negative) > 0:
#         TPR = true_positive / (true_positive + false_negative)
#     else:
#         TPR = 0
#     if (true_negative + false_positive) > 0:
#         FPR = false_positive / (true_negative + false_positive)
#     else:
#         FPR = 0
#
#     TPR_list.append(TPR)
#     FPR_list.append(FPR)
#
# AUC = pd.DataFrame(list(zip(TPR_list, FPR_list)), columns=["True Positive Rate", "False Positive Rate"])
# print(AUC.head(100))
#
# fig = px.line(AUC, x="False Positive Rate", y="True Positive Rate", width=800, height=800)
# fig.update_layout(font=dict(size=26))
# fig.show()
#
# sm = 0
# for i in range(0, len(TPR_list) - 1):
#     h = TPR_list[i]
#     sm += h * (FPR_list[i + 1] - FPR_list[i])
#
# print(sm)


"""
Plots for 21 percent of OAs in poverty
"""

# ########################################################################################################
# #
# #
# #       Classify OAs for a for a certain percentage in poverty
# #
# #
# ########################################################################################################
# percentage_in_poverty = 63
#
# pca_evs = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_eigenvectors.csv')
# print(pca_evs.head())
# print(type(pca_evs))
#
# number_in_poverty = round((len(pca_evs) / 100) * percentage_in_poverty)
#
# sorted_pca_evs = pca_evs.sort_values("Deprivation score", ascending=False)
#
# classification = []
# a = []
# count = 0
# for i in range(0, len(sorted_pca_evs)):
#     if count < number_in_poverty:
#         classification.append(1)
#         a.append("True")
#     else:
#         classification.append(0)
#         a.append("False")
#     count += 1
#
# sorted_pca_evs['classification'] = classification
# sorted_pca_evs['b'] = a
#
# ########################################################################################################
# #
# #
# #       plot data to show the classification
# #
# #
# ########################################################################################################
#
# # fig1 = px.scatter_3d(df, x="Eigenvector 2 value", y="Eigenvector 8 value", z="Eigenvector 9 value", color="classification", color_continuous_scale="Jet")
# # fig1.update_traces(marker_size=3)
# # fig1.show()
#
# ########################################################################################################
# #
# #
# #
# #
# #
# ########################################################################################################
#
# x = sorted_pca_evs[["Eigenvector 5", "Eigenvector 6", "Eigenvector 9"]].to_numpy()
# y = sorted_pca_evs["classification"].to_numpy()
#
# print(x)
# print(y)
#
# print(np.shape(x))
# print(np.shape(y))
#
# # y = list(y)
#
# print(np.shape(x))
# print(np.shape(y))
#
# clf = svm.SVC(kernel='linear', C=100000, random_state=1000)
# model = clf.fit(x, y)
#
# predictions = clf.predict(x)
# print(predictions)
#
# w = model.coef_[0]
# xx, yy = np.meshgrid(*np.array([x.min(axis=0), x.max(axis=0)])[:, :2].T)
# zz = -(w[0]/w[2])*xx - (w[1]/w[2])*yy - model.intercept_[0]/w[2]
#
#
# true_positive = 0
# false_positive = 0
# true_negative = 0
# false_negative = 0
#
# for i in range(0, len(predictions)):
#     if predictions[i] == 0 and classification[i] == 0:
#         true_negative += 1
#     elif predictions[i] == 1 and classification[i] == 1:
#         true_positive += 1
#     elif predictions[i] == 1 and classification[i] == 0:
#         false_positive += 1
#     elif predictions[i] == 0 and classification[i] == 1:
#         false_negative += 1
#
# print("true positives: ", true_positive)
# print("false positives: ", false_positive)
# print("true negatives: ", true_negative)
# print("false negatives: ", false_negative)
#
# sorted_pca_evs["predictions"] = predictions
# # df["predictions"] = df["predictions"].astype(str)
# # df["classification"] = df["classification"].astype(str)
#
# print(sorted_pca_evs.head())
# print(type(sorted_pca_evs))
#
# sorted_pca_evs.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_63_Predictions.csv', index=False)


# # Calculations for the hyperplane
# w = model.coef_[0]
# xx, yy = np.meshgrid(*np.array([x.min(axis=0), x.max(axis=0)])[:, :2].T)
# zz = -(w[0]/w[2])*xx - (w[1]/w[2])*yy - model.intercept_[0]/w[2]
#
#
# fig = px.scatter_3d(sorted_pca_evs, x="Eigenvector 5", y="Eigenvector 6", z="Eigenvector 9", color="b",
#                     hover_name="OA", range_z=[-0.05, 0.1], labels={"b": "OA classified to be in poverty"},
#                     color_continuous_scale='Bluered')
# fig.update_traces(marker_size=6)
# fig.update_layout(font=dict(size=16))
# # fig.add_traces(go.Surface(x=xx, y=yy, z=zz, opacity=.5, surfacecolor=np.zeros(zz.shape), colorscale=[[0, 'grey']]))
# fig.show()

"""
Make split and upload it 
"""
all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')
pca_predictions = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_63_Predictions.csv')

pca_predictions = pca_predictions[['OA', 'predictions']]

# variables = all_data.columns.values.tolist()
# df = pd.DataFrame(variables, columns=['Variables'])
# print(df.head())

# print(all_data.head())
# print(diffusion_map_predictions.head())
#
# print(np.shape(all_data))
# print(np.shape(diffusion_map_predictions))

splits = 2
split_sort = []

for i in range(0, len(pca_predictions)):
    OA = all_data['geography_code'][i]
    row = pca_predictions.loc[pca_predictions['OA'] == OA]
    prediction = row.iloc[0]['predictions']
    split_sort.append(prediction)

# Makes a list of the number of each OAs in each split
count = []
for i in range(0, splits):
    count.append(0)

mean_matrix = np.zeros([splits, np.shape(all_data)[1] - 1])
normalised_mean_matrix = np.zeros([splits, np.shape(all_data)[1] - 1])
all_mean_matrix = np.zeros([1, np.shape(all_data)[1] - 1])

# Finds the sum of each column for each split
for i in range(0, len(split_sort)):
    data_row = all_data.loc[i]
    data_row = data_row.to_numpy()
    data_row = np.delete(data_row, [0, 0])
    mean_matrix[split_sort[i], :] = mean_matrix[split_sort[i], :] + data_row
    all_mean_matrix = all_mean_matrix + data_row
    count[split_sort[i]] += 1

# Divides each column by the count to give the average number for each column
for i in range(0, len(count)):
    mean_matrix[i, :] = mean_matrix[i, :] / count[i]

# Gets the mean of each column
all_mean_matrix = all_mean_matrix / sum(count)


for i in range(0, splits):
    normalised_mean_matrix[i, :] = mean_matrix[i, :] / all_mean_matrix

# Do iterations for different numbers of splits
iteration = splits - 1

# Makes a list of all the variables
variables = list(all_data.columns)
variables.pop(0)
variables = np.array(variables)
variables = np.reshape(variables, [len(variables), 1])

df_set = False

while iteration > 0:
    # Do a certain number of comparisons going down by one each time
    for i in range(0, iteration):
        row_scores = normalised_mean_matrix[iteration, :] - normalised_mean_matrix[i, :]
        column_scores = np.reshape(row_scores, [len(row_scores), 1])
        variable_evaluation = np.concatenate((variables, column_scores,
                                              np.reshape(mean_matrix[iteration, :], [len(variables), 1]),
                                              np.reshape(mean_matrix[i, :], [len(variables), 1]),
                                              np.reshape(all_mean_matrix, [len(variables), 1])), axis=1)
        column_2 = str(iteration) + " - " + str(i)
        pd_variable_evaluation = pd.DataFrame(variable_evaluation, columns=["variable", column_2, str(iteration), str(i), "all_mean"])
        pd_sort_variable_evaluation = pd_variable_evaluation.sort_values(by=[column_2], ascending=False, ignore_index=True)

        if df_set:
            df = pd.concat([df, pd_sort_variable_evaluation], axis=1)
        else:
            df = pd_sort_variable_evaluation
            df_set = True

    iteration -= 1

print(df.head())

better_names = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\Variable_names_and_categories.csv')

print(better_names.head())

categories = []
display_names = []

for i in range(0, len(df)):
    variable = df['variable'][i]
    row = better_names.loc[better_names['Variables'] == variable]
    display_name = row.iloc[0]['DisplayName']
    category = row.iloc[0]['Category']
    categories.append(category)
    display_names.append(display_name)

df["categories"] = categories
df["display_names"] = display_names

print(df.head())

df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_all_63_split.csv', index=False)
