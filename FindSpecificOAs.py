import pandas as pd
import numpy as np
import math

all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\most_data.csv')
all_data_columns = all_data.columns
all_OAs = all_data['geography_code']
census_data = all_data.drop(['geography_code'], axis=1)
census_data = census_data.to_numpy()
all_data = all_data.to_numpy()


postcode_to_OA = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\PostcodeToOABristol.csv')
postcode_to_OA = postcode_to_OA.to_numpy()
bristol_OAs = postcode_to_OA[:, 1]
bristol_OAs = set(bristol_OAs)

print(np.shape(all_data)[1])

count = 0
bristol_data = np.zeros([1, np.shape(all_data)[1]])

for i in range(0, len(all_OAs)):
    if all_OAs[i] in bristol_OAs:
        count += 1
        bristol_data = np.vstack((bristol_data, all_data[i, :]))

bristol_data = np.delete(bristol_data, 0, 0)

print(count)
print(bristol_data)

bristol_data = pd.DataFrame(bristol_data, columns=all_data_columns)

bristol_data.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\BristolData.csv')

# print(type(bristol_OAs))
# print(len(bristol_OAs))
# print(bristol_OAs)



