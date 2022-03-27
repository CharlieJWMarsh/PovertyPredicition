import pandas as pd
import numpy as np
import math
import Functions as f
import heapq
import json


# all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')
#
# OAs = all_data['geography_code']
# census_data = all_data.drop(['geography_code'], axis=1)
# census_data = census_data.to_numpy()
#
# print(census_data)
#
# column_mean = np.sum(census_data, axis=0) / np.shape(census_data)[0]
# print(column_mean)
#
# column_sd = np.sqrt(np.sum((census_data - column_mean) ** 2, axis=0) / np.shape(census_data)[0])
# normalised_census_data = (census_data - column_mean) / column_sd
# faulty_columns = []
# for i in range(0, np.shape(census_data)[1]):
#     if np.isnan(np.sum(normalised_census_data[:, i])):
#         faulty_columns.insert(0, i)
# print("faulty columns: ", faulty_columns)
# for i in range(0, len(faulty_columns)):
#     normalised_census_data = np.delete(normalised_census_data, faulty_columns[i], 1)
# # np.savetxt('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Diffusion_eigenvectors\\safety_normalised_matrix.csv',
# #            normalised_census_data, delimiter=',')
# print("shape after normalise: ", np.shape(census_data))

# bristol_json = json.load(open('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\All.geojson', 'r'))
#
# useful_OAs = []
# Bris_OAs = []
#
# for i in range(0, len((bristol_json['features']))):
#     useful_OAs.append(bristol_json['features'][i]['properties']['OA11CD'])
#
# check_useful_OAs = set(useful_OAs)
#
# OAs = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs.csv')
#
# for i in range(0, len(OAs)):
#     if OAs['OA'][i] in useful_OAs:
#         Bris_OAs.append(OAs['OA'][i])
#
# print(Bris_OAs)
#
# Bris_OAs = pd.DataFrame(Bris_OAs, columns=['OA'])
# Bris_OAs.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_2.csv', index=False)

# early = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores_2011Dec.csv")
# late = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores_2018May.csv")
# df = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores_2018May.csv")
#
# count = 0
# for i in range(0, len(early['OA'])):
#     if df['score'][i] == 0:
#         count += 1
#
# print(count)

# early.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores_joint.csv', index=False)


# OAs = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs.csv')
#
# all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')
#
# bris_OA_check = set(OAs['OA'])
#
# print(bris_OA_check)

# indexs_to_remove = []
#
# for i in range(0, len(all_data)):
#     if i % 100 == 0:
#         print(i)
#     if all_data['geography_code'][i] not in bris_OA_check:
#         indexs_to_remove.insert(0, i)
#
# print(indexs_to_remove)
#
# for i in indexs_to_remove:
#     if all_data['geography_code'][i] in bris_OA_check:
#         print("fuck")
#
#     all_data = all_data.drop([i])
#     if i % 100 == 0:
#         print(i)
#
#
# all_data.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\\Input_data\\bris_OA_all_data.csv', index=False)




