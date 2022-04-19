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

all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')
diffusion_map_predictions = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\Diffusion_Map_21_Predictions.csv')

diffusion_map_predictions = diffusion_map_predictions[['OA', 'predictions']]

# variables = all_data.columns.values.tolist()
# df = pd.DataFrame(variables, columns=['Variables'])
# print(df.head())

# print(all_data.head())
# print(diffusion_map_predictions.head())
#
# print(np.shape(all_data))
# print(np.shape(diffusion_map_predictions))

splits = 2
split_sort = []

for i in range(0, len(diffusion_map_predictions)):
    OA = all_data['geography_code'][i]
    row = diffusion_map_predictions.loc[diffusion_map_predictions['OA'] == OA]
    prediction = row.iloc[0]['predictions']
    split_sort.append(prediction)

# Makes a list of the number of each OAs in each split
count = []
for i in range(0, splits):
    count.append(0)

mean_matrix = np.zeros([splits, np.shape(all_data)[1] - 1])
normalised_mean_matrix = np.zeros([splits, np.shape(all_data)[1] - 1])
all_mean_matrix = np.zeros([1, np.shape(all_data)[1] - 1])

# Finds the sum of each column for each split
for i in range(0, len(split_sort)):
    data_row = all_data.loc[i]
    data_row = data_row.to_numpy()
    data_row = np.delete(data_row, [0, 0])
    mean_matrix[split_sort[i], :] = mean_matrix[split_sort[i], :] + data_row
    all_mean_matrix = all_mean_matrix + data_row
    count[split_sort[i]] += 1

# Divides each column by the count to give the average number for each column
for i in range(0, len(count)):
    mean_matrix[i, :] = mean_matrix[i, :] / count[i]

# Gets the mean of each column
all_mean_matrix = all_mean_matrix / sum(count)


for i in range(0, splits):
    normalised_mean_matrix[i, :] = mean_matrix[i, :] / all_mean_matrix

# Do iterations for different numbers of splits
iteration = splits - 1

# Makes a list of all the variables
variables = list(all_data.columns)
variables.pop(0)
variables = np.array(variables)
variables = np.reshape(variables, [len(variables), 1])

df_set = False

while iteration > 0:
    # Do a certain number of comparisons going down by one each time
    for i in range(0, iteration):
        row_scores = normalised_mean_matrix[iteration, :] - normalised_mean_matrix[i, :]
        column_scores = np.reshape(row_scores, [len(row_scores), 1])
        variable_evaluation = np.concatenate((variables, column_scores,
                                              np.reshape(mean_matrix[iteration, :], [len(variables), 1]),
                                              np.reshape(mean_matrix[i, :], [len(variables), 1]),
                                              np.reshape(all_mean_matrix, [len(variables), 1])), axis=1)
        column_2 = str(iteration) + " - " + str(i)
        pd_variable_evaluation = pd.DataFrame(variable_evaluation, columns=["variable", column_2, str(iteration), str(i), "all_mean"])
        pd_sort_variable_evaluation = pd_variable_evaluation.sort_values(by=[column_2], ascending=False, ignore_index=True)

        if df_set:
            df = pd.concat([df, pd_sort_variable_evaluation], axis=1)
        else:
            df = pd_sort_variable_evaluation
            df_set = True

    iteration -= 1

print(df.head())

better_names = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\Variable_names_and_categories.csv')

print(better_names.head())

categories = []
display_names = []

for i in range(0, len(df)):
    variable = df['variable'][i]
    row = better_names.loc[better_names['Variables'] == variable]
    display_name = row.iloc[0]['DisplayName']
    category = row.iloc[0]['Category']
    categories.append(category)
    display_names.append(display_name)

df["categories"] = categories
df["display_names"] = display_names

print(df.head())

# df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Results\\Diffusion_map_all_21_split.csv', index=False)
