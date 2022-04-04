import plotly.express as px
import pandas as pd
import json


# Load in geojson file
bristol_json = json.load(open('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\All.geojson', 'r'))


# geo_df = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Analysis\\RFRegression_test_predictions.csv")

# Deprivation scores
geo_df = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\OAs_with_scores.csv")

# Best heatmap score
# geo_df = pd.read_csv("C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Heatmap_data_2_normalised.csv")


fig = px.choropleth_mapbox(geo_df, locations="OA", color="score", featureidkey="properties.OA11CD", geojson=bristol_json,
                           mapbox_style="carto-positron", center={"lat":51.481951, "lon":-2.526448}, zoom=13)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

fig.show()
