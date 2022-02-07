import numpy as np


def find_large_sd_away_row(all_data, OA, threshold, if_print):
    census_data_pandas = all_data.drop(['geography_code'], axis=1)
    census_data = census_data_pandas.to_numpy()

    column_mean = np.sum(census_data, axis=0) / np.shape(census_data)[0]
    column_sd = np.sqrt(np.sum((census_data - column_mean) ** 2, axis=0) / np.shape(census_data)[0])

    column_mean = column_mean.reshape(1, len(column_mean))
    column_sd = column_sd.reshape(1, len(column_sd))

    investigated_row = all_data.loc[all_data['geography_code'] == OA]
    investigated_row = investigated_row.drop(['geography_code'], axis=1)
    investigated_row = investigated_row.to_numpy()

    difference_from_average = abs(investigated_row - column_mean)
    standard_deviations_away = difference_from_average / column_sd

    interesting_index = []
    interesting_column = []
    interesting_column_sd_away = []

    for i in range(0, np.shape(standard_deviations_away)[1]):
        if standard_deviations_away[0, i] > threshold:
            interesting_index.append(i)

    for i in interesting_index:
        interesting_column.append(census_data_pandas.columns[i])
        interesting_column_sd_away.append(standard_deviations_away[0, i])
        if if_print:
            print(census_data_pandas.columns[i], "\nsd away: ", standard_deviations_away[0, i], "\n")

    return interesting_column, interesting_column_sd_away


def find_large_sd_away_multiple_rows(all_data, OAs, threshold_sd, threshold_OAs, if_print):
    common_interesting_columns = []
    count_interesting_columns = []
    avg_sd_away_interesting_columns = []

    threshold_common_interesting_columns = []
    threshold_count_interesting_columns = []
    threshold_avg_sd_away_interesting_columns = []

    for i in range(0, len(OAs)):
        columns, sd_away = find_large_sd_away_row(all_data, OAs[i], threshold_sd, False)

        for j in range(0, len(columns)):
            if columns[j] not in common_interesting_columns:
                common_interesting_columns.append(columns[j])
                count_interesting_columns.append(1)
                avg_sd_away_interesting_columns.append(sd_away[j])
            else:
                index = common_interesting_columns.index(columns[j])
                avg_sd_away_interesting_columns[index] = ((avg_sd_away_interesting_columns[index] *
                                                           count_interesting_columns[index]) + sd_away[j]) / \
                                                         (count_interesting_columns[index] + 1)
                count_interesting_columns[index] = count_interesting_columns[index] + 1

    for i in range(0, len(count_interesting_columns)):
        if count_interesting_columns[i] >= threshold_OAs:
            threshold_common_interesting_columns.append(common_interesting_columns[i])
            threshold_count_interesting_columns.append(count_interesting_columns[i])
            threshold_avg_sd_away_interesting_columns.append(avg_sd_away_interesting_columns[i])
            if if_print:
                print('\n', common_interesting_columns[i], "\nColumn found in ", count_interesting_columns[i], " OAs",
                      "\nAvg sd away: ", avg_sd_away_interesting_columns[i])

    return threshold_common_interesting_columns, threshold_count_interesting_columns, threshold_avg_sd_away_interesting_columns
