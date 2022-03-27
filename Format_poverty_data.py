import pandas as pd
import numpy as np


OAs = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_2.csv')

all_mappings = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OA_to_all_2018May.csv')

scores_mappings = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Poverty_data\\LSOA_2010_scores.csv')

# print(OAs.head())
# print(all_mappings.head())
# print(scores_mappings.head())
#
# print(np.shape(OAs))

scores = []
ISOA_check_set = set(scores_mappings['LSOA_CODE'])
OA_check_set = set(all_mappings['oa11cd'])

# OA = OAs['OA'][51]
# print(OA)
#
# index_of_OA = all_mappings[all_mappings['oa11cd'] == OA].index.values[0]
# print(index_of_OA)
#
# ISOA = all_mappings['lsoa11cd'][index_of_OA]
# print(ISOA)
#
# if ISOA in ISOA_check_set:
#     index_of_ISOA = scores_mappings[scores_mappings['LSOA_CODE'] == ISOA].index.values[0]
#     print(index_of_ISOA)
#
#     score = scores_mappings['IMD SCORE'][index_of_ISOA]
#     print(score)
#
#     scores.append(score)
#     print(scores)
# else:
#     print('Missing score for ', ISOA)
#     scores.append(0)


for i in range(0, np.shape(OAs)[0]):
    OA = OAs['OA'][i]
    # print(OA)

    if OA in OA_check_set:
        index_of_OA = all_mappings[all_mappings['oa11cd'] == OA].index.values[0]
        # print(index_of_OA)

        ISOA = all_mappings['lsoa11cd'][index_of_OA]
        # print(ISOA)

        if ISOA in ISOA_check_set:
            index_of_ISOA = scores_mappings[scores_mappings['LSOA_CODE'] == ISOA].index.values[0]
            # print(index_of_ISOA)

            score = scores_mappings['IMD SCORE'][index_of_ISOA]
            # print(score)

            scores.append(score)
            # print(scores)
        else:
            print('Missing score for ', ISOA)
            scores.append(0)

    else:
        print('Missing OA ', OA, ' for mapping')
        scores.append(0)

    if i % 100 == 0:
        print(i)


OAs['score'] = scores

OAs.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores_2018May_2.csv', index=False)


