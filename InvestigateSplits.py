import pandas as pd
import numpy as np
import Analysis_functions as af
import statistics
import plotly.express as px
from sklearn import svm


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)


# df = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Analysis\\EV2FPs.csv')
#
# FP_0_to_20 = []
# FP_20_to_40 = []
# FP_40_plus = []
# FN_0_to_20 = []
# FN_20_to_40 = []
# FN_40_plus = []
# P_0_to_20 = []
# P_20_to_40 = []
# P_40_plus = []
#
# for i in range(0, len(df)):
#     if df["DeprivaionScore"][i] < 20:
#         if df["False negative classifications"][i] > 19:
#             FN_0_to_20.append(df["OA"][i])
#         elif df["False positive classifications"][i] > 19:
#             FP_0_to_20.append(df["OA"][i])
#         else:
#             P_0_to_20.append(df["OA"][i])
#
#     elif df["DeprivaionScore"][i] < 40:
#         if df["False negative classifications"][i] > 19:
#             FN_20_to_40.append(df["OA"][i])
#         elif df["False positive classifications"][i] > 19:
#             FP_20_to_40.append(df["OA"][i])
#         else:
#             P_20_to_40.append(df["OA"][i])
#
#     else:
#         if df["False negative classifications"][i] > 24:
#             FN_40_plus.append(df["OA"][i])
#         elif df["False positive classifications"][i] > 24:
#             FP_40_plus.append(df["OA"][i])
#         else:
#             P_40_plus.append(df["OA"][i])
#
#
# # print(len(FP_0_to_20))
# # print(len(FP_20_to_40))
# # print(len(FP_40_plus))
# # print('\n')
# # print(len(FN_0_to_20))
# # print(len(FN_20_to_40))
# # print(len(FN_40_plus))
# # print('\n')
# # print(len(P_0_to_20))
# # print(len(P_20_to_40))
# # print(len(P_40_plus))
#
# all_0_to_20 = FP_0_to_20 + FN_0_to_20 + P_0_to_20
# all_20_to_40 = FP_20_to_40 + FN_20_to_40 + P_20_to_40
# all_40_plus = FP_40_plus + FN_40_plus + P_40_plus
#
# all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')
#
# df_initiated = False
# for i in range(0, len(all_40_plus)):
#     OA = all_40_plus[i]
#     row = df.loc[df['OA'] == OA]
#
#     if df_initiated:
#         group_data = group_data.append(row, ignore_index=True)
#     else:
#         group_data = row
#         df_initiated = True
#
# print(group_data.head(20))
# print(np.shape(group_data))
#
# group_data.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Analysis\\EV2FPs_Deptivation_40_plus.csv', index=False)

"""
"""

# all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data_Deprivation_20_to_40.csv')
# false_positives = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Analysis\\EV2FPs_Deprivation_20_to_40.csv')
#
#
# splits = 3
# split_sort = []
#
# # Makes a list of split values
# for i in range(0, len(false_positives)):
#     if false_positives["False positive classifications"][i] > 19:
#         split_sort.append(0)
#     elif false_positives["False negative classifications"][i] > 19:
#         split_sort.append(2)
#     else:
#         split_sort.append(1)
#
# # print(split_sort)
# # print(np.shape(all_data))
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
#
# print(df.head(20))
#
# df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Analysis\\EV2_Deprivation_20_to_40_comparison.csv', index=False)

"""

"""

evaluate_0_to_20 = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Analysis\\EV2_Deprivation_0_to_20_comparison.csv')
evaluate_20_to_40 = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Analysis\\EV2_Deprivation_20_to_40_comparison.csv')

# print(evaluate_20_to_40.head())

false_positives_0_to_20 = evaluate_0_to_20[["variable.2", "1 - 0", "1.1", "0.1", "all_mean.2"]]
false_negatives_0_to_20 = evaluate_0_to_20[["variable.1", "2 - 1", "2.1", "1", "all_mean.1"]]
false_negatives_20_to_40 = evaluate_20_to_40[["variable.1", "2 - 1", "2.1", "1", "all_mean.1"]]

false_positives_0_to_20.loc[:, "1 - 0"] *= -1
# print(false_positives_0_to_20.head(10))

overall_scores = false_positives_0_to_20[["variable.2", "1 - 0"]]
# print(overall_scores.head(10))

for i in range(0, len(overall_scores)):
    variable = overall_scores["variable.2"][i]

    value_1 = false_negatives_0_to_20.loc[false_negatives_0_to_20['variable.1'] == variable]
    value_1 = value_1.iloc[0]["2 - 1"]

    value_2 = false_negatives_20_to_40.loc[false_negatives_20_to_40['variable.1'] == variable]
    value_2 = value_2.iloc[0]["2 - 1"]
    overall_scores["1 - 0"][i] = overall_scores["1 - 0"][i] + value_1 + value_2


overall_scores = overall_scores.sort_values(by="1 - 0", ascending=False, ignore_index=True)

useful_overall_scores = overall_scores.loc[[1]]
# print(useful_overall_scores)

for i in range(2, len(overall_scores)):
    variable = overall_scores["variable.2"][i]
    value = false_positives_0_to_20.loc[false_positives_0_to_20['variable.2'] == variable]
    value = value.iloc[0]["all_mean.2"]
    if value > 1:
        row = overall_scores.loc[[i]]
        useful_overall_scores = useful_overall_scores.append(row)

useful_overall_scores = useful_overall_scores.head(20)
useful_overall_scores = useful_overall_scores.reset_index()
#
# print(useful_overall_scores.head(40))
#
# fig1 = px.bar(useful_overall_scores, x='variable.2', y='1 - 0', labels={'variable.2': "Variables", '1 - 0': "Joint standard deviations away"})
# fig1.update_layout(font=dict(size=20))
# fig1.show()
# # print(useful_overall_scores['variable.2'])


"""
Plot eigenvectors against score
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
# # print(df.head())
#
# all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')
# all_data = all_data.rename(columns={"geography_code": "OA"})
#
# score = [0] * len(all_data)
#
# for j in range(0, len(useful_overall_scores)):
#     var = useful_overall_scores["variable.2"][j]
#     column = all_data.loc[:, str(var)]
#     column_as_list = list(column)
#
#     list_mean = statistics.mean(column_as_list)
#     list_sd = statistics.stdev(column_as_list)
#     for i in range(0, len(column_as_list)):
#         column_as_list[i] = (column_as_list[i] - list_mean) / list_sd
#
#     score = [x + y for x, y in zip(score, column_as_list)]
#
# print(score)
#
# for i in range(0, len(score)):
#     if score[i] > 150:
#         score[i] = 5
#
# df["Poorly classified variables combined score"] = score
#
# fig1 = px.scatter(df, hover_name="OA", x="Poorly classified variables combined score", y="Eigenvector 1 value", color="Deprivation score")
# fig1.update_layout(font=dict(size=32))
# fig1.show()
#
# fig2 = px.scatter(df, hover_name="OA", x="Poorly classified variables combined score", y="Eigenvector 3 value", color="Deprivation score")
# fig2.update_layout(font=dict(size=32))
# fig2.show()
#
# fig3 = px.scatter(df, hover_name="OA", x="Poorly classified variables combined score", y="Eigenvector 4 value", color="Deprivation score")
# fig3.update_layout(font=dict(size=32))
# fig3.show()
#
# fig4 = px.scatter(df, hover_name="OA", x="Poorly classified variables combined score", y="Eigenvector 5 value", color="Deprivation score")
# fig4.update_layout(font=dict(size=32))
# fig4.show()
#
# fig5 = px.scatter(df, hover_name="OA", x="Poorly classified variables combined score", y="Eigenvector 6 value", color="Deprivation score")
# fig5.update_layout(font=dict(size=32))
# fig5.show()
#
# fig6 = px.scatter(df, hover_name="OA", x="Poorly classified variables combined score", y="Eigenvector 7 value", color="Deprivation score")
# fig6.update_layout(font=dict(size=32))
# fig6.show()
#
# fig7 = px.scatter(df, hover_name="OA", x="Poorly classified variables combined score", y="Eigenvector 8 value", color="Deprivation score")
# fig7.update_layout(font=dict(size=32))
# fig7.show()
#
# fig8 = px.scatter(df, hover_name="OA", x="Poorly classified variables combined score", y="Eigenvector 9 value", color="Deprivation score")
# fig8.update_layout(font=dict(size=32))
# fig8.show()
#
# fig9 = px.scatter(df, hover_name="OA", x="Poorly classified variables combined score", y="Eigenvector 10 value", color="Deprivation score")
# fig9.update_layout(font=dict(size=32))
# fig9.show()
