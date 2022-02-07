import pandas as pd
import numpy as np
import Analysis_functions as af


all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')

OAs = ['E00073583', 'E00073173', 'E00073154', 'E00073264']

threshold_sd = 5

threshold_OAs = 4


columns, count_of_OAs, avg_sd_away = af.find_large_sd_away_multiple_rows(all_data, OAs, threshold_sd, threshold_OAs, True)

print('\n')
print(columns)
print(count_of_OAs)
print(avg_sd_away)

# common_interesting_columns = []
# count_interesting_columns = []
# avg_sd_away_interesting_columns = []
#
# threshold_common_interesting_columns = []
# threshold_count_interesting_columns = []
# threshold_avg_sd_away_interesting_columns = []
#
# for i in range(0, len(OAs)):
#     columns, sd_away = af.find_large_sd_away_row(all_data, OAs[i], threshold_sd)
#
#     for j in range(0, len(columns)):
#         if columns[j] not in common_interesting_columns:
#             common_interesting_columns.append(columns[j])
#             count_interesting_columns.append(1)
#             avg_sd_away_interesting_columns.append(sd_away[j])
#         else:
#             index = common_interesting_columns.index(columns[j])
#             avg_sd_away_interesting_columns[index] = ((avg_sd_away_interesting_columns[index] *
#                                                        count_interesting_columns[index]) + sd_away[j]) / \
#                                                      (count_interesting_columns[index] + 1)
#             count_interesting_columns[index] = count_interesting_columns[index] + 1
#
# for i in range(0, len(count_interesting_columns)):
#     if count_interesting_columns[i] >= threshold_OAs:
#         threshold_common_interesting_columns.append(common_interesting_columns[i])
#         threshold_count_interesting_columns.append(count_interesting_columns[i])
#         threshold_avg_sd_away_interesting_columns.append(avg_sd_away_interesting_columns[i])
#         print('\n', common_interesting_columns[i], "\nColumn found in ", count_interesting_columns[i], " OAs", "\nAvg sd away: ",
#               avg_sd_away_interesting_columns[i])
#
# print(threshold_common_interesting_columns)
# print(threshold_count_interesting_columns)
# print(threshold_avg_sd_away_interesting_columns)






