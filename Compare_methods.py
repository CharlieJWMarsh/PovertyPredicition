import pandas as pd
import numpy as np
import Analysis_functions as af
import statistics
import plotly.express as px
from sklearn import svm
import plotly.graph_objects as go


pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
Load in datasets
"""
# Random forest
df_RFR = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\RF_all_split.csv")
# print(df_RFR.head())

# Diffusion maps
df_DM = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\Diffusion_map_all_21_split.csv")
df_DM = df_DM.rename(columns={"1 - 0": "Importance"})
# print(df_DM.head())

# PCA
df_PCA = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\PCA_all_21_split.csv")
df_PCA = df_PCA.rename(columns={"1 - 0": "Importance"})
# print(df_PCA.head())

all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')

poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")

"""
Look at 2% groups of certain variables 
"""

# poverty_score_data = poverty_score_data.sort_values(by='score')
# # print(poverty_score_data.head())
#
# groups = 50
# groups_left = 50
# group_labels = []
# group_count = [0] * groups
# oas_left = len(poverty_score_data)
# # print(oas_left)
#
#
# for i in range(0, groups_left):
#     for j in range(0, int(round(oas_left / groups_left))):
#         group_labels.append(i)
#         group_count[i] += 1
#     oas_left -= int(round(oas_left / groups_left))
#     groups_left -= 1
#
# # print(len(group_labels))
#
# poverty_score_data["classification"] = group_labels
#
# # print(poverty_score_data)
#
# people_NS_SeC_never_worked_long_term_unemployed_list = [0] * groups
# household_social_rented_rented_from_council_local_authority_list = [0] * groups
# people_ecominically_inactive_long_term_sick_or_disabled_list = [0] * groups
# households_no_cars_no_vans_list = [0] * groups
# total_cars_or_vans_in_area_list = [0] * groups
#
# for i in range(0, len(poverty_score_data)):
#     OA = poverty_score_data["OA"][i]
#     classification = poverty_score_data["classification"][i]
#     row = all_data.loc[all_data['geography_code'] == OA]
#
#     people_NS_SeC_never_worked_long_term_unemployed_list[classification] += row.iloc[0][
#         "people_NS_SeC_never_worked_long_term_unemployed"]
#     household_social_rented_rented_from_council_local_authority_list[classification] += row.iloc[0][
#         "household_social_rented_rented_from_council_local_authority"]
#     people_ecominically_inactive_long_term_sick_or_disabled_list[classification] += row.iloc[0][
#         "people_ecominically_inactive_long_term_sick_or_disabled"]
#     households_no_cars_no_vans_list[classification] += row.iloc[0]["households_no_cars_no_vans"]
#     total_cars_or_vans_in_area_list[classification] += row.iloc[0]["total_cars_or_vans_in_area"]
#
# print(people_NS_SeC_never_worked_long_term_unemployed_list)
# # print(household_social_rented_rented_from_council_local_authority_list)
# # print(people_ecominically_inactive_long_term_sick_or_disabled_list)
# # print(households_no_cars_no_vans_list)
# # print(total_cars_or_vans_in_area_list)
#
#
# people_NS_SeC_never_worked_long_term_unemployed_mean = [0] * groups
# household_social_rented_rented_from_council_local_authority_mean = [0] * groups
# people_ecominically_inactive_long_term_sick_or_disabled_mean = [0] * groups
# households_no_cars_no_vans_mean = [0] * groups
# total_cars_or_vans_in_area_mean = [0] * groups
#
# for i in range(0, groups):
#     people_NS_SeC_never_worked_long_term_unemployed_mean[i] = people_NS_SeC_never_worked_long_term_unemployed_list[i] / group_count[i]
#     household_social_rented_rented_from_council_local_authority_mean[i] = household_social_rented_rented_from_council_local_authority_list[i] / group_count[i]
#     people_ecominically_inactive_long_term_sick_or_disabled_mean[i] = people_ecominically_inactive_long_term_sick_or_disabled_list[i] / group_count[i]
#     households_no_cars_no_vans_mean[i] = households_no_cars_no_vans_list[i] / group_count[i]
#     total_cars_or_vans_in_area_mean[i] = total_cars_or_vans_in_area_list[i] / group_count[i]
#
# print(people_NS_SeC_never_worked_long_term_unemployed_mean)
#
# x = [0] * groups
# size_of_each_group = 100 / groups
# starting_group = size_of_each_group / 2
# for i in range(0, groups):
#     x[i] = starting_group + (i * size_of_each_group)
#
# print(x)
#
# df = pd.DataFrame({'Percentile of Deprivation data': x,
#                    'people_NS_SeC_never_worked_long_term_unemployed': people_NS_SeC_never_worked_long_term_unemployed_mean,
#                    'household_social_rented_rented_from_council_local_authority': household_social_rented_rented_from_council_local_authority_mean,
#                    'people_ecominically_inactive_long_term_sick_or_disabled': people_ecominically_inactive_long_term_sick_or_disabled_mean,
#                    'households_no_cars_no_vans': households_no_cars_no_vans_mean,
#                    'total_cars_or_vans_in_area': total_cars_or_vans_in_area_mean
#                    })
#
# fig = px.line(df, x="Percentile of Deprivation data", y=df.columns[1:6], labels={'value': 'Mean Value per OA'})
# fig.update_layout(font=dict(size=18))
# fig.show()

"""
Compare DM and PCA
"""

# # print(df_DM.head())
# # print(df_PCA.head())
#
# variable_list = []
# importance_list = []
#
# # variable = df_DM["variable"][0]
# # importance_dm = df_DM["Importance"][0]
# # row = df_PCA.loc[df_PCA['variable'] == variable]
# # importance_pca = row.iloc[0]["Importance"]
# #
# # importance_diff = importance_dm - importance_pca
#
# # print(importance_diff)
#
# for i in range(0, len(df_DM)):
#     variable = df_DM["variable"][i]
#     importance_dm = df_DM["Importance"][i]
#     row = df_PCA.loc[df_PCA['variable'] == variable]
#     importance_pca = row.iloc[0]["Importance"]
#
#     importance_diff = importance_dm - importance_pca
#
#     variable_list.append(variable)
#     importance_list.append(importance_diff)
#
# df = pd.DataFrame({
#     "Variable": variable_list, "Difference": importance_list
# })
#
# df = df.sort_values(by='Difference', ascending=False)
# print(df)
# df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\Difference_between_PCA_and_DM.csv', index=False)

"""

"""

poverty_score_data = poverty_score_data.sort_values(by='score')
# print(poverty_score_data.head())

groups = 25
groups_left = 25
group_labels = []
group_count = [0] * groups
oas_left = len(poverty_score_data)


for i in range(0, groups_left):
    for j in range(0, int(round(oas_left / groups_left))):
        group_labels.append(i)
        group_count[i] += 1
    oas_left -= int(round(oas_left / groups_left))
    groups_left -= 1

poverty_score_data["classification"] = group_labels

var1_list = [0] * groups
var2_list = [0] * groups
var3_list = [0] * groups
var4_list = [0] * groups
var5_list = [0] * groups

for i in range(0, len(poverty_score_data)):
    OA = poverty_score_data["OA"][i]
    classification = poverty_score_data["classification"][i]
    row = all_data.loc[all_data['geography_code'] == OA]

    var1_list[classification] += row.iloc[0][
        "area_hectares"]
    var2_list[classification] += row.iloc[0][
        "males_industry_agriculture_forestry_fishing"]
    var3_list[classification] += row.iloc[0][
        "people_industry_agriculture_forestry_fishing"]
    var4_list[classification] += row.iloc[0]["females_industry_agriculture_forestry_fishing"]
    var5_list[classification] += row.iloc[0]["caravan_or_other_mobile_or_temporary_structure"]

# print(var1_list)
# print(var2_list)
# print(var3_list)
# print(var4_list)
# print(var5_list)


var1_mean = [0] * groups
var2_mean = [0] * groups
var3_mean = [0] * groups
var4_mean = [0] * groups
var5_mean = [0] * groups

for i in range(0, groups):
    var1_mean[i] = var1_list[i] / group_count[i]
    var2_mean[i] = var2_list[i] / group_count[i]
    var3_mean[i] = var3_list[i] / group_count[i]
    var4_mean[i] = var4_list[i] / group_count[i]
    var5_mean[i] = var5_list[i] / group_count[i]

x = [0] * groups
size_of_each_group = 100 / groups
starting_group = size_of_each_group / 2
for i in range(0, groups):
    x[i] = starting_group + (i * size_of_each_group)

print(x)

df = pd.DataFrame({'Percentile of Deprivation data': x,
                   'var1': var1_mean,
                   'var2': var2_mean,
                   'var3': var3_mean,
                   'var4': var4_mean,
                   'var5': var5_mean
                   })

fig = px.line(df, x="Percentile of Deprivation data", y=df.columns[1], labels={'var1': 'Mean Value of area_hectares per OA'})
fig.update_layout(font=dict(size=18))
fig.show()
