from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import statistics
import math
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
import pydot
import plotly.express as px


"""
Sort out datasets to feed to Random Forest
"""
all_data = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Input_data\\all_data.csv')
poverty_score_data = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")

# Takes data from csv without OAs
features = all_data.drop(['geography_code'], axis=1)
feature_list = list(features.columns)
features = features.to_numpy()


# print(features[1, :])
print(np.shape(features))

score = poverty_score_data.drop(['OA'], axis=1)
score = score.to_numpy()

# print(score[1, :])
print(np.shape(score))


"""
Split data into train test split
"""
# Split the data into training and testing sets
train_features, test_features, train_score, test_score = train_test_split(features, score, test_size=0.3,
                                                                          random_state=32456)

train_score = train_score.reshape((len(train_score)))
test_score = test_score.reshape((len(test_score)))

print('Train Features Shape:', train_features.shape)
print('Train Score Shape:', train_score.shape)
print('Test Features Shape:', test_features.shape)
print('Test Score Shape:', test_score.shape)


"""
Train Random Forest and perform predictions on the test data
"""
rf = RandomForestRegressor(n_estimators=1000,
                           max_depth=20,
                           min_samples_split=25,
                           min_samples_leaf=12,
                           min_impurity_decrease=0,
                           max_leaf_nodes=1000,
                           n_jobs=-1,
                           random_state=32456)
rf.fit(train_features, train_score)

predictions = rf.predict(test_features)


"""
Calculate baseline scores where a mean score is predicted for each OA 
"""
# Finds baseline linear error
mean_score = np.mean(test_score)
baseline_error = abs(mean_score - test_score)
print('\n')
print('Baseline Absolute Error: ', np.mean(baseline_error))

# Calculate mean absolute percentage error (MAPE)
baseline_mape = 100 * (baseline_error / test_score)
baseline_accuracy = 100 - np.mean(baseline_mape)
print('Accuracy:', round(baseline_accuracy, 2), '%.')

# Finds baseline MSE
mean = np.full((len(test_score),), np.mean(test_score))
baseline_mean_squared_error = mean_squared_error(mean, test_score)
print('Baseline MSE: ', baseline_mean_squared_error)


"""
Calculate the scores for trained model
"""
# Calculate the absolute errors
errors = abs(predictions - test_score)
print('\n')
print('Mean Absolute Error:', np.mean(errors))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_score)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Finds MSE
mean_squared_error = mean_squared_error(predictions, test_score)
print('Mean Squared Error: ', mean_squared_error)


"""
Make a diagram of a tree used in the random forest
"""
# Pull out one tree from the forest
tree = rf.estimators_[4]
# Export the image to a dot file
export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


"""
Find the most important features and their importance value
"""
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 10)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# Print out the feature and importances
# print('\n')
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# Saves a csv of features importance
df = pd.DataFrame(feature_importances, columns=["Feature", "Importance"])
print('\n')
print(df.head(10))
# df.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Analysis\\RFRegression_Feature_Importance.csv', index=False)


"""
Plot bar chart of most important features for random forest 
"""
features_to_plot = df.head(20)
fig1 = px.bar(features_to_plot, x='Feature', y='Importance')
fig1.show()






"""
Predict all of the test OAs score and format for heatmap
"""
# Split the data into training and testing sets
train_features, test_features, train_poverty_score_data, test_poverty_score_data = train_test_split(features,
                                                                                                    poverty_score_data,
                                                                                                    test_size=0.3,
                                                                                                    random_state=32456)

test_OAs = test_poverty_score_data['OA']
test_OAs = test_OAs.to_numpy()

test_OAs = np.reshape(test_OAs, (len(test_OAs), 1))
predictions = np.reshape(predictions, (len(predictions), 1))

predicted_OAs = np.concatenate((test_OAs, predictions), axis=1)

df2 = pd.DataFrame(predicted_OAs, columns=['OA', 'score'])
print(df2.head())

# df2.to_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Analysis\\RFRegression_test_predictions.csv', index=False)


"""
Plot OA test against OA deprivation score 
"""
deprivation_scores = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")
print(df2.head())
print(df2['OA'][0])
OA = df2['OA'][0]
print(OA)
print(deprivation_scores.head())
print(deprivation_scores.loc[deprivation_scores['OA'] == OA])
score = float(deprivation_scores.loc[deprivation_scores['OA'] == OA]['score'])
print(score)


deprivation_score_test_OAs = []
for i in range(0, len(df2)):
    OA = df2['OA'][i]
    deprivation_score_test_OAs.append(float(deprivation_scores.loc[deprivation_scores['OA'] == OA]['score']))

print(len(deprivation_score_test_OAs))
print(deprivation_score_test_OAs)

df2['Depriavtion score'] = deprivation_score_test_OAs
print(df2.head())
df2 = df2.rename(columns={'score': 'Poverty score'})


fig2 = px.scatter(df2, x='Poverty score', y='Depriavtion score')
fig2.add_shape(type="line", x0=0, y0=0, x1=60, y1=60, line=dict(
        color="Red",
        width=2,
    ))
fig2.show()


