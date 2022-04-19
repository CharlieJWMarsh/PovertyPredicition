import pandas as pd
import numpy as np
import Analysis_functions as af
import statistics
import plotly.express as px


pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

"""
code for looking out stand out variables in a small number of OAs
"""
# all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')
#
# OAs = ['E00073583', 'E00073173', 'E00073154', 'E00073264']
#
# threshold_sd = 5
#
# threshold_OAs = 4
#
# columns, count_of_OAs, avg_sd_away = af.find_large_sd_away_multiple_rows(all_data, OAs, threshold_sd, threshold_OAs, False)
#
# print('\n')
# print(columns)
# print(count_of_OAs)
# print(avg_sd_away)

"""
finds average standard deviation away from values in each OA for diffusion map eigenvector and deprivation data
"""
# heatmap_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv")
# poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
#
# heatmap_scores = heatmap_data["score"]
# heatmap_scores = heatmap_scores.to_list()
# poverty_scores = poverty_score_data["score"]
# poverty_scores = poverty_scores.to_list()
#
# mean_heatmap_scores = sum(heatmap_scores) / len(heatmap_scores)
# mean_poverty_scores = sum(poverty_scores) / len(poverty_scores)
# sd_heatmap_scores = statistics.stdev(heatmap_scores)
# sd_poverty_scores = statistics.stdev(poverty_scores)
#
# for i in range(0, len(heatmap_scores)):
#     heatmap_scores[i] = (heatmap_scores[i] - mean_heatmap_scores) / sd_heatmap_scores
#     poverty_scores[i] = (poverty_scores[i] - mean_poverty_scores) / sd_poverty_scores
#
#
# total_error = 0
#
# for i in range(0, len(heatmap_scores)):
#     total_error += abs(heatmap_scores[i] - poverty_scores[i])
#
# avg_error = total_error / len(heatmap_scores)
#
# print(avg_error)
#
# heatmap_data["score"] = heatmap_scores
# poverty_score_data["score"] = poverty_scores
#
# poverty_score_data.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\\OA_plot_data\\OAs_with_scores_normalised.csv', index=False)
# heatmap_data.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2_normalised.csv', index=False)


"""
plots histogram of 
"""
heatmap_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2_normalised.csv")

heatmap_scores = heatmap_data["score"]
heatmap_scores = heatmap_scores.to_list()

fig = px.histogram(heatmap_scores, nbins=200)
fig.show()

"""
Compares the splits of the diffusion map and makes it into a excel file 
"""
# heatmap_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2_normalised.csv")
# all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')
#
# splits = 3
# splits_range = [[-500, 0.4], [0.4, 1.98], [1.98, 500]]
# split_sort = []
#
# # Makes a list of split values
# for i in range(0, len(heatmap_data)):
#     for j in range(0, splits):
#         if splits_range[j][1] > heatmap_data["score"][i] > splits_range[j][0]:
#             split_sort.append(j)
#
# print(split_sort)
# print(np.shape(all_data))
#
# # Makes a list of the number of each OAs in each split
# count = []
# for i in range(0, splits):
#     count.append(0)
#
# mean_matrix = np.zeros([splits, np.shape(all_data)[1] - 1])
# normalised_mean_matrix = np.zeros([splits, np.shape(all_data)[1] - 1])
# all_mean_matrix = np.zeros([1, np.shape(all_data)[1] - 1])
#
#
# print(mean_matrix)
# print(np.shape(mean_matrix))
# print(np.shape(all_mean_matrix))
#
#
# # Finds the sum of each column for each split
# for i in range(0, len(split_sort)):
#     data_row = all_data.loc[i]
#     data_row = data_row.to_numpy()
#     data_row = np.delete(data_row, [0, 0])
#     mean_matrix[split_sort[i], :] = mean_matrix[split_sort[i], :] + data_row
#     all_mean_matrix = all_mean_matrix + data_row
#     count[split_sort[i]] += 1
#
# print(mean_matrix)
# print(count)
# print(sum(count))
#
#
# # Divides each column by the count to give the average number for each column
# for i in range(0, len(count)):
#     mean_matrix[i, :] = mean_matrix[i, :] / count[i]
#
# # Gets the mean of each column
# all_mean_matrix = all_mean_matrix / sum(count)
#
# print(mean_matrix)
# print(all_mean_matrix[0, 0:5])
#
# for i in range(0, splits):
#     normalised_mean_matrix[i, :] = mean_matrix[i, :] / all_mean_matrix
#
# print(normalised_mean_matrix)
#
# # Do iterations for different numbers of splits
# iteration = splits - 1
#
# # Makes a list of all the variables
# variables = list(all_data.columns)
# variables.pop(0)
# variables = np.array(variables)
# variables = np.reshape(variables, [len(variables), 1])
# print(np.shape(variables))
# print(len(variables))
#
# df_set = False
#
# while iteration > 0:
#     # Do a certain number of comparisons going down by one each time
#     for i in range(0, iteration):
#         row_scores = normalised_mean_matrix[iteration, :] - normalised_mean_matrix[i, :]
#         column_scores = np.reshape(row_scores, [len(row_scores), 1])
#         variable_evaluation = np.concatenate((variables, column_scores,
#                                               np.reshape(mean_matrix[iteration, :], [len(variables), 1]),
#                                               np.reshape(mean_matrix[i, :], [len(variables), 1]),
#                                               np.reshape(all_mean_matrix, [len(variables), 1])), axis=1)
#         # variable_evaluation[:, 0] = variables
#         # variable_evaluation[:, 1] = column_scores
#         # variable_evaluation[:, 2] = np.reshape(mean_matrix[iteration, :], [len(variables), 1])
#         # variable_evaluation[:, 3] = np.reshape(mean_matrix[i, :], [len(variables), 1])
#         print(variable_evaluation)
#         column_2 = str(iteration) + " - " + str(i)
#         pd_variable_evaluation = pd.DataFrame(variable_evaluation, columns=["variable", column_2, str(iteration), str(i), "all_mean"])
#         print(pd_variable_evaluation.head())
#         pd_sort_variable_evaluation = pd_variable_evaluation.sort_values(by=[column_2], ascending=False, ignore_index=True)
#         print(pd_sort_variable_evaluation.head())
#
#         if df_set:
#             df = pd.concat([df, pd_sort_variable_evaluation], axis=1)
#         else:
#             df = pd_sort_variable_evaluation
#             df_set = True
#
#         print(df.head())
#
#     iteration -= 1

# df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Analysis\\Poverty_comparison.csv', index=False)

"""
Plot a scatter graph of the data of the data against poverty data
"""
# heatmap_data_1 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_1.csv")
# heatmap_data_2 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv")
# heatmap_data_3 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_3.csv")
# heatmap_data_4 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_4.csv")
# heatmap_data_5 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_5.csv")
# heatmap_data_6 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_6.csv")
# heatmap_data_7 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_7.csv")
# heatmap_data_8 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_8.csv")
# heatmap_data_9 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_9.csv")
# heatmap_data_10 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_10.csv")
# poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
#
# df_1 = pd.merge(heatmap_data_1, poverty_score_data, on='OA')
# df_1 = df_1.rename(columns={"score_x": "Eigenvector 1 value", "score_y": "Deprivation score"})
# df_2 = pd.merge(heatmap_data_2, poverty_score_data, on='OA')
# df_2 = df_2.rename(columns={"score_x": "Eigenvector 2 value", "score_y": "Deprivation score"})
# df_3 = pd.merge(heatmap_data_3, poverty_score_data, on='OA')
# df_3 = df_3.rename(columns={"score_x": "Eigenvector 3 value", "score_y": "Deprivation score"})
# df_4 = pd.merge(heatmap_data_4, poverty_score_data, on='OA')
# df_4 = df_4.rename(columns={"score_x": "Eigenvector 4 value", "score_y": "Deprivation score"})
# df_5 = pd.merge(heatmap_data_5, poverty_score_data, on='OA')
# df_5 = df_5.rename(columns={"score_x": "Eigenvector 5 value", "score_y": "Deprivation score"})
# df_6 = pd.merge(heatmap_data_6, poverty_score_data, on='OA')
# df_6 = df_6.rename(columns={"score_x": "Eigenvector 6 value", "score_y": "Deprivation score"})
# df_7 = pd.merge(heatmap_data_7, poverty_score_data, on='OA')
# df_7 = df_7.rename(columns={"score_x": "Eigenvector 7 value", "score_y": "Deprivation score"})
# df_8 = pd.merge(heatmap_data_8, poverty_score_data, on='OA')
# df_8 = df_8.rename(columns={"score_x": "Eigenvector 8 value", "score_y": "Deprivation score"})
# df_9 = pd.merge(heatmap_data_9, poverty_score_data, on='OA')
# df_9 = df_9.rename(columns={"score_x": "Eigenvector 9 value", "score_y": "Deprivation score"})
# df_10 = pd.merge(heatmap_data_10, poverty_score_data, on='OA')
# df_10 = df_10.rename(columns={"score_x": "Eigenvector 10 value", "score_y": "Deprivation score"})
#
# fig1 = px.scatter(df_1, x="Deprivation score", y="Eigenvector 1 value", trendline="ols")
# fig2 = px.scatter(df_2, hover_name="OA", x="Deprivation score", y="Eigenvector 2 value", trendline="ols", width=1200, height=800)
# fig3 = px.scatter(df_3, x="Deprivation score", y="Eigenvector 3 value", trendline="ols")
# fig4 = px.scatter(df_4, x="Deprivation score", y="Eigenvector 4 value", trendline="ols")
# fig5 = px.scatter(df_5, x="Deprivation score", y="Eigenvector 5 value", trendline="ols")
# fig6 = px.scatter(df_6, x="Deprivation score", y="Eigenvector 6 value", trendline="ols")
# fig7 = px.scatter(df_7, x="Deprivation score", y="Eigenvector 7 value", trendline="ols")
# fig8 = px.scatter(df_8, x="Deprivation score", y="Eigenvector 8 value", trendline="ols")
# fig9 = px.scatter(df_9, x="Deprivation score", y="Eigenvector 9 value", trendline="ols")
# fig10 = px.scatter(df_10, x="Deprivation score", y="Eigenvector 10 value", trendline="ols")
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
Compare 2 eigenvectors at once to see if there is a non linear pattern in the data
"""

# heatmap_data_1 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_1.csv")
# heatmap_data_2 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv")
# heatmap_data_3 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_3.csv")
# heatmap_data_4 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_4.csv")
# heatmap_data_5 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_5.csv")
# heatmap_data_6 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_6.csv")
# heatmap_data_7 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_7.csv")
# heatmap_data_8 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_8.csv")
# heatmap_data_9 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_9.csv")
# heatmap_data_10 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_10.csv")
# poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
#
#
# df = pd.merge(heatmap_data_1, heatmap_data_2, on='OA')
# df = df.rename(columns={"score_x": "Eigenvector 1 value", "score_y": "Eigenvector 2 value"})
# df = pd.merge(df, heatmap_data_3, on='OA')
# df = df.rename(columns={"score": "Eigenvector 3 value"})
# df = pd.merge(df, heatmap_data_4, on='OA')
# df = df.rename(columns={"score": "Eigenvector 4 value"})
# df = pd.merge(df, heatmap_data_5, on='OA')
# df = df.rename(columns={"score": "Eigenvector 5 value"})
# df = pd.merge(df, heatmap_data_6, on='OA')
# df = df.rename(columns={"score": "Eigenvector 6 value"})
# df = pd.merge(df, heatmap_data_7, on='OA')
# df = df.rename(columns={"score": "Eigenvector 7 value"})
# df = pd.merge(df, heatmap_data_8, on='OA')
# df = df.rename(columns={"score": "Eigenvector 8 value"})
# df = pd.merge(df, heatmap_data_9, on='OA')
# df = df.rename(columns={"score": "Eigenvector 9 value"})
# df = pd.merge(df, heatmap_data_10, on='OA')
# df = df.rename(columns={"score": "Eigenvector 10 value"})
# df = pd.merge(df, poverty_score_data, on='OA')
# df = df.rename(columns={"score": "Deprivation score"})
#
#
# fig1 = px.scatter(df, x="Eigenvector 2 value", y="Eigenvector 5 value", color="Deprivation score")
# fig2 = px.scatter(df, x="Eigenvector 2 value", y="Eigenvector 6 value", color="Deprivation score")
# fig3 = px.scatter(df, x="Eigenvector 2 value", y="Eigenvector 7 value", color="Deprivation score")
# fig4 = px.scatter(df, x="Eigenvector 2 value", y="Eigenvector 8 value", color="Deprivation score")
# fig5 = px.scatter(df, x="Eigenvector 2 value", y="Eigenvector 9 value", color="Deprivation score")
# fig6 = px.scatter(df, x="Eigenvector 2 value", y="Eigenvector 8 value", color="Deprivation score")
# fig7 = px.scatter(df, x="Eigenvector 2 value", y="Eigenvector 9 value", color="Deprivation score")
# fig8 = px.scatter(df, x="Eigenvector 2 value", y="Eigenvector 9 value", color="Deprivation score")
# fig9 = px.scatter(df, x="Eigenvector 2 value", y="Eigenvector 2 value", color="Deprivation score")


# fig1.show()
# fig2.show()
# fig3.show()
# fig4.show()
# fig5.show()
# fig6.show()
# fig7.show()
# fig8.show()
# fig9.show()

"""
Compare 3 eigenvectors at once to see if there is a non linear pattern in the data
"""
# heatmap_data_1 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_1.csv")
# heatmap_data_2 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv")
# heatmap_data_3 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_3.csv")
# heatmap_data_4 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_4.csv")
# heatmap_data_5 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_5.csv")
# heatmap_data_6 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_6.csv")
# heatmap_data_7 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_7.csv")
# heatmap_data_8 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_8.csv")
# heatmap_data_9 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_9.csv")
# heatmap_data_10 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_10.csv")
# poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
#
#
# df = pd.merge(heatmap_data_1, heatmap_data_2, on='OA')
# df = df.rename(columns={"score_x": "Eigenvector 1 value", "score_y": "Eigenvector 2 value"})
# df = pd.merge(df, heatmap_data_3, on='OA')
# df = df.rename(columns={"score": "Eigenvector 3 value"})
# df = pd.merge(df, heatmap_data_4, on='OA')
# df = df.rename(columns={"score": "Eigenvector 4 value"})
# df = pd.merge(df, heatmap_data_5, on='OA')
# df = df.rename(columns={"score": "Eigenvector 5 value"})
# df = pd.merge(df, heatmap_data_6, on='OA')
# df = df.rename(columns={"score": "Eigenvector 6 value"})
# df = pd.merge(df, heatmap_data_7, on='OA')
# df = df.rename(columns={"score": "Eigenvector 7 value"})
# df = pd.merge(df, heatmap_data_8, on='OA')
# df = df.rename(columns={"score": "Eigenvector 8 value"})
# df = pd.merge(df, heatmap_data_9, on='OA')
# df = df.rename(columns={"score": "Eigenvector 9 value"})
# df = pd.merge(df, heatmap_data_10, on='OA')
# df = df.rename(columns={"score": "Eigenvector 10 value"})
# df = pd.merge(df, poverty_score_data, on='OA')
# df = df.rename(columns={"score": "Deprivation score"})
#
#
# fig1 = px.scatter_3d(df, x="Eigenvector 2 value", y="Eigenvector 8 value", z="Eigenvector 9 value", color="Deprivation score")
# fig1.update_traces(marker_size=3)
# fig2 = px.scatter_3d(df, x="Eigenvector 9 value", y="Eigenvector 2 value", z="Eigenvector 3 value", color="Deprivation score")
# fig2.update_traces(marker_size=3)
# fig3 = px.scatter_3d(df, x="Eigenvector 9 value", y="Eigenvector 2 value", z="Eigenvector 4 value", color="Deprivation score")
# fig3.update_traces(marker_size=3)
# fig4 = px.scatter_3d(df, x="Eigenvector 9 value", y="Eigenvector 2 value", z="Eigenvector 6 value", color="Deprivation score")
# fig4.update_traces(marker_size=3)
# fig5 = px.scatter_3d(df, x="Eigenvector 9 value", y="Eigenvector 2 value", z="Eigenvector 7 value", color="Deprivation score")
# fig5.update_traces(marker_size=3)
# fig6 = px.scatter_3d(df, x="Eigenvector 9 value", y="Eigenvector 2 value", z="Eigenvector 8 value", color="Deprivation score")
# fig6.update_traces(marker_size=3)
# fig7 = px.scatter_3d(df, x="Eigenvector 9 value", y="Eigenvector 2 value", z="Eigenvector 9 value", color="Deprivation score")
# fig7.update_traces(marker_size=3)
# fig8 = px.scatter_3d(df, x="Eigenvector 9 value", y="Eigenvector 2 value", z="Eigenvector 10 value", color="Deprivation score")
# fig8.update_traces(marker_size=3)


# fig1.show()
# fig2.show()
# fig3.show()
# fig4.show()
# fig5.show()
# fig6.show()
# fig7.show()
# fig8.show()

"""
Use linear correlation coefficients for classification
"""
# # heatmap_data_1 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_1.csv")
# heatmap_data_2 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv")
# heatmap_data_3 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_3.csv")
# # heatmap_data_4 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_4.csv")
# # heatmap_data_5 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_5.csv")
# heatmap_data_6 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_6.csv")
# # heatmap_data_7 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_7.csv")
# # heatmap_data_8 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_8.csv")
# # heatmap_data_9 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_9.csv")
# # heatmap_data_10 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_10.csv")
# poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
#
# # coefficients = [-0.046028453, 0.794468631, 0.139204757, 0.063054356, -0.11795421, 0.195552539, 0.071638709,
# #                 -0.107291144, -0.048045617, 0.055400434]
# coefficients = [0.703552, 0.123274, 0.173174]
#
# score = np.zeros([len(heatmap_data_2), 3])
# scores = np.zeros([len(heatmap_data_2), 1])
#
# # score[:, 0] = heatmap_data_1["score"]
# # score[:, 1] = heatmap_data_2["score"]
# # score[:, 2] = heatmap_data_3["score"]
# # score[:, 3] = heatmap_data_4["score"]
# # score[:, 4] = heatmap_data_5["score"]
# # score[:, 5] = heatmap_data_6["score"]
# # score[:, 6] = heatmap_data_7["score"]
# # score[:, 7] = heatmap_data_8["score"]
# # score[:, 8] = heatmap_data_9["score"]
# # score[:, 9] = heatmap_data_10["score"]
#
# score[:, 0] = heatmap_data_2["score"]
# score[:, 1] = heatmap_data_3["score"]
# score[:, 2] = heatmap_data_6["score"]
#
# for i in range(0, len(heatmap_data_2)):
#     scores[i, 0] = sum(score[i, :] * coefficients)
#
# scores = pd.DataFrame(scores, columns=['Eigenvector score'])
# print(scores)
# print(poverty_score_data)
#
#
# df = pd.concat([scores, poverty_score_data["score"]], axis=1)
# df = df.rename(columns={"score": "Deprivation score"})
#
# fig = px.scatter(df, x="Deprivation score", y="Eigenvector score", trendline="ols")
#
# fig.show()
