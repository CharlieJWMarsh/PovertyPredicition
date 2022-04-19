import pandas as pd
import numpy as np
import Analysis_functions as af
import statistics
import plotly.express as px
from sklearn import svm


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# percentage_in_poverty = 27
#
# poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
#
# number_in_poverty = round((len(poverty_score_data) / 100) * percentage_in_poverty)
#
# sorted_poverty_score = poverty_score_data.sort_values("score", ascending=False)
#
# classification = []
# count = 0
# for i in range(0, len(poverty_score_data)):
#     if count < number_in_poverty:
#         classification.append(1)
#     else:
#         classification.append(0)
#     count += 1
#
# sorted_poverty_score['classification'] = classification
#
# heatmap_data_2 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv")
#
# df_2 = pd.merge(heatmap_data_2, sorted_poverty_score, on='OA')
# df_2 = df_2.rename(columns={"score_x": "Eigenvector 2 value", "score_y": "Deprivation score"})
#
# fig2 = px.scatter(df_2, hover_name="OA", x="Deprivation score", y="Eigenvector 2 value", color="classification", trendline="ols")
# fig2.show()

# x = df_2["Eigenvector 2 value"].to_numpy()
# y = df_2["classification"].to_numpy()
#
# print(x)
# print(y)
#
# print(np.shape(x))
# print(np.shape(y))
#
# x = np.reshape(x, (len(x), 1))
# y = list(y)
#
# print(np.shape(x))
# print(np.shape(y))
#
# clf = svm.SVC(kernel='linear')
# clf.fit(x, y)
#
# predictions = clf.predict(x)
# print(predictions)
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


"""
Comparing accuracy of prediction by looking at classification by percentage
"""

# poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
# heatmap_data_2 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv")
#
# TPR_list = []
# FPR_list = []
#
# for j in range(1, 99):
#     number_in_poverty = round((len(poverty_score_data) / 100) * j)
#
#     # print('\n')
#     # print("Number of OAs: ", len(poverty_score_data))
#     # print("Percent in poverty: ", j)
#     # print("Number in poverty: ", number_in_poverty)
#
#     classification = []
#     count = 0
#     for i in range(0, len(poverty_score_data)):
#         if count < number_in_poverty:
#             classification.append(1)
#         else:
#             classification.append(0)
#         count += 1
#
#     df = pd.merge(poverty_score_data, heatmap_data_2, on='OA')
#     df = df.rename(columns={"score_x": "DeprivaionScore", "score_y": "EV2Value"})
#
#     df = df.sort_values("DeprivaionScore", ascending=False)
#     df['DeprivationClassification'] = classification
#
#     df = df.sort_values("EV2Value", ascending=False)
#     df['EV2Classification'] = classification
#
#     df = df.sort_index()
#
#     true_positive = 0
#     false_positive = 0
#     true_negative = 0
#     false_negative = 0
#
#     for i in range(0, len(df)):
#         if df["DeprivationClassification"][i] == 0 and df["EV2Classification"][i] == 0:
#             true_negative += 1
#         elif df["DeprivationClassification"][i] == 1 and df["EV2Classification"][i] == 1:
#             true_positive += 1
#         elif df["DeprivationClassification"][i] == 1 and df["EV2Classification"][i] == 0:
#             false_positive += 1
#         elif df["DeprivationClassification"][i] == 0 and df["EV2Classification"][i] == 1:
#             false_negative += 1
#
#     # print('\n')
#     # print("true positives: ", true_positive)
#     # print("false positives: ", false_positive)
#     # print("true negatives: ", true_negative)
#     # print("false negatives: ", false_negative)
#
#     TPR = true_positive / (true_positive + false_negative)
#     FPR = false_positive / (true_negative + false_positive)
#
#     # print('\n')
#     # print("True Positive Rate: ", TPR)
#     # print("False Positive Rate: ", FPR)
#
#     TPR_list.append(TPR)
#     FPR_list.append(FPR)
#
# # print(TPR_list)
# # print(FPR_list)
#
# AUC = pd.DataFrame(list(zip(TPR_list, FPR_list)), columns=["True Positive Rate", "False Positive Rate"])
# print(AUC.head(100))
# print(AUC.to_string())
#
# fig = px.line(AUC, x="False Positive Rate", y="True Positive Rate", width=800, height=800)
# fig.show()
#
# sm = 0
# for i in range(0, len(TPR_list) - 1):
#     h = TPR_list[i]
#     sm += h * (FPR_list[i + 1] - FPR_list[i])
#
# print(sm)

"""
Find number of false predicitions
"""

# poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
# heatmap_data_2 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv")
#
# total_false_predictions = [0] * len(poverty_score_data)
# print(len(total_false_predictions))
#
# for j in range(1, 99):
#     number_in_poverty = round((len(poverty_score_data) / 100) * j)
#
#     # print('\n')
#     # print("Number of OAs: ", len(poverty_score_data))
#     # print("Percent in poverty: ", j)
#     # print("Number in poverty: ", number_in_poverty)
#
#     classification = []
#     count = 0
#     for i in range(0, len(poverty_score_data)):
#         if count < number_in_poverty:
#             classification.append(1)
#         else:
#             classification.append(0)
#         count += 1
#
#     df = pd.merge(poverty_score_data, heatmap_data_2, on='OA')
#     df = df.rename(columns={"score_x": "DeprivaionScore", "score_y": "EV2Value"})
#
#     df = df.sort_values("DeprivaionScore", ascending=False)
#     df['DeprivationClassification'] = classification
#
#     df = df.sort_values("EV2Value", ascending=False)
#     df['EV2Classification'] = classification
#
#     df = df.sort_index()
#
#     for i in range(0, len(df)):
#         if df["DeprivationClassification"][i] == 1 and df["EV2Classification"][i] == 0:
#             total_false_predictions[i] += 1
#         elif df["DeprivationClassification"][i] == 0 and df["EV2Classification"][i] == 1:
#             total_false_predictions[i] += 1
#
# # print(total_false_predictions)
#
# df["False classifications"] = total_false_predictions
#
# print(df.head(30))
#
# fig = px.scatter(df, hover_name="OA", x="DeprivaionScore", y="EV2Value", color="False classifications", trendline="ols")
# fig.show()

"""
Find number of false positives and negatives
"""

poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
heatmap_data_2 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv")

total_false_positive_predictions = [0] * len(poverty_score_data)
print(len(total_false_positive_predictions))
total_false_negative_predictions = [0] * len(poverty_score_data)
print(len(total_false_positive_predictions))

for j in range(1, 99):
    number_in_poverty = round((len(poverty_score_data) / 100) * j)

    # print('\n')
    # print("Number of OAs: ", len(poverty_score_data))
    # print("Percent in poverty: ", j)
    # print("Number in poverty: ", number_in_poverty)

    classification = []
    count = 0
    for i in range(0, len(poverty_score_data)):
        if count < number_in_poverty:
            classification.append(1)
        else:
            classification.append(0)
        count += 1

    df = pd.merge(poverty_score_data, heatmap_data_2, on='OA')
    df = df.rename(columns={"score_x": "DeprivaionScore", "score_y": "EV2Value"})

    df = df.sort_values("DeprivaionScore", ascending=False)
    df['DeprivationClassification'] = classification

    df = df.sort_values("EV2Value", ascending=False)
    df['EV2Classification'] = classification

    df = df.sort_index()

    for i in range(0, len(df)):
        if df["DeprivationClassification"][i] == 1 and df["EV2Classification"][i] == 0:
            total_false_negative_predictions[i] += 1
        elif df["DeprivationClassification"][i] == 0 and df["EV2Classification"][i] == 1:
            total_false_positive_predictions[i] += 1

# print(total_false_predictions)

df["False negative classifications"] = total_false_negative_predictions
df["False positive classifications"] = total_false_positive_predictions

print(df.head(30))

fig1 = px.scatter(df, hover_name="OA", x="DeprivaionScore", y="EV2Value", color="False negative classifications", trendline="ols")
fig1.show()
fig2 = px.scatter(df, hover_name="OA", x="DeprivaionScore", y="EV2Value", color="False positive classifications", trendline="ols")
fig2.show()

df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Analysis\\EV2FPs.csv', index=False)


"""
Plot true false positive and negatives for certain thresholds
"""

# poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
# heatmap_data_2 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv")
#
# percentage_in_poverty = 30
#
# number_in_poverty = round((len(poverty_score_data) / 100) * percentage_in_poverty)
#
# # print('\n')
# # print("Number of OAs: ", len(poverty_score_data))
# # print("Percent in poverty: ", j)
# # print("Number in poverty: ", number_in_poverty)
#
# classification = []
# count = 0
# for i in range(0, len(poverty_score_data)):
#     if count < number_in_poverty:
#         classification.append(1)
#     else:
#         classification.append(0)
#     count += 1
#
# df = pd.merge(poverty_score_data, heatmap_data_2, on='OA')
# df = df.rename(columns={"score_x": "DeprivaionScore", "score_y": "EV2Value"})
#
# df = df.sort_values("DeprivaionScore", ascending=False)
# df['DeprivationClassification'] = classification
#
# df = df.sort_values("EV2Value", ascending=False)
# df['EV2Classification'] = classification
#
# df = df.sort_index()
#
# state = []
#
# for i in range(0, len(df)):
#     if df["DeprivationClassification"][i] == 0 and df["EV2Classification"][i] == 0:
#         state.append(0)
#     elif df["DeprivationClassification"][i] == 1 and df["EV2Classification"][i] == 1:
#         state.append(1)
#     elif df["DeprivationClassification"][i] == 1 and df["EV2Classification"][i] == 0:
#         state.append(2)
#     elif df["DeprivationClassification"][i] == 0 and df["EV2Classification"][i] == 1:
#         state.append(3)
#
# df["state"] = state
#
# print(df.head())
#
# fig = px.scatter(df, hover_name="OA", x="DeprivaionScore", y="EV2Value", color="state", trendline="ols")
# fig.show()
