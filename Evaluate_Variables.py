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
Load in datasets
"""
# Random forest
# df = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\RF_all_split.csv")
# print(df.head())

# Diffusion maps
df = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\Diffusion_map_all_21_split.csv")
df = df.rename(columns={"1 - 0": "Importance"})
print(df.head())

# PCA
# df = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_all_21_split.csv")
# df = df.rename(columns={"1 - 0": "Importance"})
# print(df.head())


"""
Plot stuff for categories 
"""

# econ_total = 0
# social_total = 0
# environment_total = 0
# demographic_total = 0
#
# econ_count = 0
# social_count = 0
# environment_count = 0
# demographic_count = 0
#
# all_econ_count = 0
# all_social_count = 0
# all_environment_count = 0
# all_demographic_count = 0
#
# all_econ_total = 0
# all_social_total = 0
# all_environment_total = 0
# all_demographic_total = 0
#
# df2 = df.head(88)
# # df3 = df.head(337)
#
# for i in range(0, len(df)):
#     value = df["Importance"][i]
#     if df["categories"][i] == "Economic":
#         all_econ_total += value
#         all_econ_count += 1
#     if df["categories"][i] == "Social":
#         all_social_total += value
#         all_social_count += 1
#     if df["categories"][i] == "Environmental":
#         all_environment_total += value
#         all_environment_count += 1
#     if df["categories"][i] == "Demographic":
#         all_demographic_total += value
#         all_demographic_count += 1
#
# for i in range(0, len(df2)):
#     value = df2["Importance"][i]
#     if df2["categories"][i] == "Economic":
#         econ_total += value
#         econ_count += 1
#     if df2["categories"][i] == "Social":
#         social_total += value
#         social_count += 1
#     if df2["categories"][i] == "Environmental":
#         environment_total += value
#         environment_count += 1
#     if df2["categories"][i] == "Demographic":
#         demographic_total += value
#         demographic_count += 1
#
# sum_all_count = all_econ_count + all_social_count + all_environment_count + all_demographic_count
#
# all_econ_count = round((all_econ_count / sum_all_count) * 100)
# all_social_count = round((all_social_count / sum_all_count) * 100)
# all_environment_count = round((all_environment_count / sum_all_count) * 100)
# all_demographic_count = round((all_demographic_count / sum_all_count) * 100)
#
# print('\n')
# print(all_econ_count)
# print(all_social_count)
# print(all_environment_count)
# print(all_demographic_count)
#
# # sum_all_total = all_econ_total + all_social_total + all_environment_total + all_demographic_total
# #
# # all_econ_total = round((all_econ_total / sum_all_total) * 100)
# # all_social_total = round((all_social_total / sum_all_total) * 100)
# # all_environment_total = round((all_environment_total / sum_all_total) * 100)
# # all_demographic_total = round((all_demographic_total / sum_all_total) * 100)
# #
# # all_econ_mean = all_econ_total / all_econ_count
# # all_social_mean = all_social_total / all_social_count
# # all_environment_mean = all_environment_total / all_environment_count
# # all_demographic_mean = all_demographic_total / all_demographic_count
# #
# # sum_all_mean = all_econ_mean + all_social_mean + all_environment_mean + all_demographic_mean
# #
# # all_econ_mean = round((all_econ_mean / sum_all_mean) * 100)
# # all_social_mean = round((all_social_mean / sum_all_mean) * 100)
# # all_environment_mean = round((all_environment_mean / sum_all_mean) * 100)
# # all_demographic_mean = round((all_demographic_mean / sum_all_mean) * 100)
# #
# # print('\n')
# # print(all_econ_mean)
# # print(all_social_mean)
# # print(all_environment_mean)
# # print(all_demographic_mean)
#
#
#
# sum_total = econ_total + social_total + environment_total + demographic_total
#
# econ_total = round((econ_total / sum_total) * 100)
# social_total = round((social_total / sum_total) * 100)
# environment_total = round((environment_total / sum_total) * 100)
# demographic_total = round((demographic_total / sum_total) * 100)
#
# print('\n')
# print(econ_total)
# print(social_total)
# print(environment_total)
# print(demographic_total)
#
# sum_count = econ_count + social_count + environment_count + demographic_count
#
# econ_count = round((econ_count / sum_count) * 100)
# social_count = round((social_count / sum_count) * 100)
# environment_count = round((environment_count / sum_count) * 100)
# demographic_count = round((demographic_count / sum_count) * 100)
#
# print('\n')
# print(econ_count)
# print(social_count)
# print(environment_count)
# print(demographic_count)
#
# econ_mean = econ_total / econ_count
# social_mean = social_total / social_count
# environment_mean = environment_total / environment_count
# demographic_mean = demographic_total / demographic_count
#
# sum_mean = econ_mean + social_mean + environment_mean + demographic_mean
#
# econ_mean = round((econ_mean / sum_mean) * 100)
# social_mean = round((social_mean / sum_mean) * 100)
# environment_mean = round((environment_mean / sum_mean) * 100)
# demographic_mean = round((demographic_mean / sum_mean) * 100)
#
# print('\n')
# print(econ_mean)
# print(social_mean)
# print(environment_mean)
# print(demographic_mean)
#
# x = ['Number of variables', 'Number of significant variables', 'Cumulative significance of significant variables',
#      'Mean variable significance of significant variables']
#
# plot = go.Figure(data=[go.Bar(
#     name='Economic',
#     x=x,
#     y=[all_econ_count, econ_count, econ_total, econ_mean]
#    ),
#                        go.Bar(
#     name='Social',
#     x=x,
#     y=[all_social_count, social_count, social_total, social_mean]
#    ),
#                        go.Bar(
#     name='Environmental',
#     x=x,
#     y=[all_environment_count, environment_count, environment_total, environment_mean]
#    ),
#                        go.Bar(
#     name='Demographic',
#     x=x,
#     y=[all_demographic_count, demographic_count, demographic_total, demographic_mean]
#    )
# ])

# x = ['Number of variables', 'Number of significant variables', 'Cumulative significance',
#      'Cumulative significance of significant variables', 'Mean variable significance',
#      'Mean variable significance of significant variables']
#
# plot = go.Figure(data=[go.Bar(
#     name='Economic',
#     x=x,
#     y=[all_econ_count, econ_count, all_econ_total, econ_total, all_econ_mean, econ_mean]
#    ),
#                        go.Bar(
#     name='Social',
#     x=x,
#     y=[all_social_count, social_count, all_social_total, social_total, all_social_mean, social_mean]
#    ),
#                        go.Bar(
#     name='Environmental',
#     x=x,
#     y=[all_environment_count, environment_count, all_environment_total, environment_total, all_environment_mean, environment_mean]
#    ),
#                        go.Bar(
#     name='Demographic',
#     x=x,
#     y=[all_demographic_count, demographic_count, all_demographic_total, demographic_total, all_demographic_mean, demographic_mean]
#    )
# ])
#
# plot.update_layout(barmode='stack', xaxis=dict(title=""), yaxis=dict(title="Percentage"), font=dict(size=20))
#
# plot.show()

"""
Plot most important features
"""

df = df.head(10)
# df = df.tail(20)

fig = px.bar(df, y='display_names', x='Importance', labels={'display_names': "Variables", 'Importance': "Feature importance"}, width=1200, height=600)
fig.update_layout(font=dict(size=14))
fig.show()

"""
Split each Variable into their class
"""

# social_df = df.loc[df["categories"] == "Social"]
# # social_df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_21_Predictions_Social.csv', index=False)
#
# economic_df = df.loc[df["categories"] == "Economic"]
# # economic_df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_Predictions_Economic.csv', index=False)
#
# environmental_df = df.loc[df["categories"] == "Environmental"]
# # environmental_df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_Predictions_Environmental.csv', index=False)
#
# demographic_df = df.loc[df["categories"] == "Demographic"]
# # demographic_df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_21_Predictions_Demographic.csv', index=False)

# social_df = social_df.head(20)
# fig1 = px.bar(social_df, y='display_names', x='Importance', labels={'display_names': "Variables", 'Importance': "Feature importance"})
# fig1.update_layout(font=dict(size=22))
# fig1.show()
#
# economic_df = economic_df.head(20)
# fig2 = px.bar(economic_df, y='display_names', x='Importance', labels={'display_names': "Variables", 'Importance': "Feature importance"})
# fig2.update_layout(font=dict(size=22))
# fig2.show()
#
# environmental_df = environmental_df.head(20)
# fig3 = px.bar(environmental_df, y='display_names', x='Importance', labels={'display_names': "Variables", 'Importance': "Feature importance"})
# fig3.update_layout(font=dict(size=22))
# fig3.show()
#
# demographic_df = demographic_df.head(20)
# fig4 = px.bar(demographic_df, y='display_names', x='Importance', labels={'display_names': "Variables", 'Importance': "Feature importance"})
# fig4.update_layout(font=dict(size=22))
# fig4.show()
#
# all_df = df.head(20)
# fig5 = px.bar(all_df, y='display_names', x='Importance', labels={'display_names': "Variables", 'Importance': "Feature importance"})
# fig5.update_layout(font=dict(size=22))
# fig5.show()
#
# fig6 = px.line(df, x='Count', y='Importance', width=800, height=800,
#                labels={'Count': "Count of Variables", 'Importance': "Feature importance"})
# fig6.update_layout(font=dict(size=22))
# # fig6.update_layout(xaxis={'visible': False, 'showticklabels': False})
# fig6.show()


