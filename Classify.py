import pandas as pd
import numpy as np
import Analysis_functions as af
import statistics
import plotly.express as px
from sklearn import svm
import plotly.graph_objects as go



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
Use SVM to classify a single percentage
"""

########################################################################################################
#
#
#       Classify OAs for a for a certain percentage in poverty
#
#
########################################################################################################
percentage_in_poverty = 63

poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")

number_in_poverty = round((len(poverty_score_data) / 100) * percentage_in_poverty)

sorted_poverty_score = poverty_score_data.sort_values("score", ascending=False)

classification = []
a = []
count = 0
for i in range(0, len(poverty_score_data)):
    if count < number_in_poverty:
        classification.append(1)
        a.append("True")
    else:
        classification.append(0)
        a.append("False")
    count += 1

sorted_poverty_score['classification'] = classification
sorted_poverty_score['b'] = a


########################################################################################################
#
#
#       Load in data and make the dataset
#
#
########################################################################################################

heatmap_data_2 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv")
heatmap_data_8 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_8.csv")
heatmap_data_9 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_9.csv")

df = pd.merge(sorted_poverty_score, heatmap_data_2, on='OA')
df = df.rename(columns={"score_y": "Eigenvector 2 value", "score_x": "Deprivation score"})

df = pd.merge(df, heatmap_data_8, on='OA')
df = df.rename(columns={"score": "Eigenvector 8 value"})

df = pd.merge(df, heatmap_data_9, on='OA')
df = df.rename(columns={"score": "Eigenvector 9 value"})

print(df.head())

########################################################################################################
#
#
#       plot data to show the classification
#
#
########################################################################################################

# fig1 = px.scatter_3d(df, x="Eigenvector 2 value", y="Eigenvector 8 value", z="Eigenvector 9 value", color="classification", color_continuous_scale="Jet")
# fig1.update_traces(marker_size=3)
# fig1.show()

########################################################################################################
#
#
#
#
#
########################################################################################################

x = df[["Eigenvector 2 value", "Eigenvector 8 value", "Eigenvector 9 value"]].to_numpy()
y = df["classification"].to_numpy()

print(x)
print(y)

print(np.shape(x))
print(np.shape(y))

# y = list(y)

print(np.shape(x))
print(np.shape(y))

clf = svm.SVC(kernel='linear', C=100000, random_state=1000)
model = clf.fit(x, y)

predictions = clf.predict(x)
print(predictions)

w = model.coef_[0]
xx, yy = np.meshgrid(*np.array([x.min(axis=0), x.max(axis=0)])[:, :2].T)
zz = -(w[0]/w[2])*xx - (w[1]/w[2])*yy - model.intercept_[0]/w[2]


true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

for i in range(0, len(predictions)):
    if predictions[i] == 0 and classification[i] == 0:
        true_negative += 1
    elif predictions[i] == 1 and classification[i] == 1:
        true_positive += 1
    elif predictions[i] == 1 and classification[i] == 0:
        false_positive += 1
    elif predictions[i] == 0 and classification[i] == 1:
        false_negative += 1

print("true positives: ", true_positive)
print("false positives: ", false_positive)
print("true negatives: ", true_negative)
print("false negatives: ", false_negative)

df["predictions"] = predictions
# df["predictions"] = df["predictions"].astype(str)
# df["classification"] = df["classification"].astype(str)

print(df.head())
print(type(df))

df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\Diffusion_Map_63_Predictions.csv', index=False)


# # Calculations for the hyperplane
# w = model.coef_[0]
# xx, yy = np.meshgrid(*np.array([x.min(axis=0), x.max(axis=0)])[:, :2].T)
# zz = -(w[0]/w[2])*xx - (w[1]/w[2])*yy - model.intercept_[0]/w[2]
#
#
# fig = px.scatter_3d(df, x="Eigenvector 2 value", y="Eigenvector 8 value", z="Eigenvector 9 value", color="b",
#                     hover_name="OA", range_z=[-0.05, 0.1], labels={"b": "OA predicted to be in poverty"},
#                     color_continuous_scale='Bluered')
# fig.update_traces(marker_size=6)
# fig.update_layout(font=dict(size=16))
# # fig.add_traces(go.Surface(x=xx, y=yy, z=zz, opacity=.5, surfacecolor=np.zeros(zz.shape), colorscale=[[0, 'grey']]))
# fig.show()

"""
Use SVM to classify all percentages
"""

# ########################################################################################################
# #
# #
# #       Classify OAs for a for a certain percentage in poverty
# #
# #
# ########################################################################################################
#
# poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
#
# TPR_list = []
# FPR_list = []
#
# for i in range(1, 99):
#
#     number_in_poverty = round((len(poverty_score_data) / 100) * i)
#
#     sorted_poverty_score = poverty_score_data.sort_values("score", ascending=False)
#
#     classification = []
#     count = 0
#     for j in range(0, len(poverty_score_data)):
#         if count < number_in_poverty:
#             classification.append(1)
#         else:
#             classification.append(0)
#         count += 1
#
#     sorted_poverty_score['classification'] = classification
#
#     ########################################################################################################
#     #
#     #
#     #       Load in data and make the dataset
#     #
#     #
#     ########################################################################################################
#
#     heatmap_data_2 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2.csv")
#     heatmap_data_8 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_8.csv")
#     heatmap_data_9 = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_9.csv")
#
#     df = pd.merge(sorted_poverty_score, heatmap_data_2, on='OA')
#     df = df.rename(columns={"score_y": "Eigenvector 2 value", "score_x": "Deprivation score"})
#
#     df = pd.merge(df, heatmap_data_8, on='OA')
#     df = df.rename(columns={"score": "Eigenvector 8 value"})
#
#     df = pd.merge(df, heatmap_data_9, on='OA')
#     df = df.rename(columns={"score": "Eigenvector 9 value"})
#
#     ########################################################################################################
#     #
#     #
#     #       plot data to show the classification
#     #
#     #
#     ########################################################################################################
#
#     # fig1 = px.scatter_3d(df, x="Eigenvector 2 value", y="Eigenvector 8 value", z="Eigenvector 9 value", color="classification", color_continuous_scale="Jet")
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
#     x = df[["Eigenvector 2 value", "Eigenvector 8 value", "Eigenvector 9 value"]].to_numpy()
#     y = df["classification"].to_numpy()
#
#     clf = svm.SVC(kernel='linear', C=100000, random_state=1000)
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
# fig = px.line(AUC, x="False Positive Rate", y="True Positive Rate", width=1000, height=800)
# fig.update_layout(font=dict(size=26))
# fig.show()
#
# sm = 0
# for i in range(0, len(TPR_list) - 1):
#     h = TPR_list[i]
#     sm += h * (FPR_list[i + 1] - FPR_list[i])
#
# print(sm)
