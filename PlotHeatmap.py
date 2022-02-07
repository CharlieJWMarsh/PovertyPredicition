import plotly.express as px
import pandas as pd
import json


# Load in geojson file
bristol_json = json.load(open('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\OA_plot_data\\Bristol_OA.geojson', 'r'))

# Load in eigenvector
geo_df = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Heatmaps\\Bristolheatmap_alldata_2.csv')

fig = px.choropleth_mapbox(geo_df, locations="OA", color="score", featureidkey="properties.oa11cd", geojson=bristol_json,
                           mapbox_style="carto-positron", center={"lat":51.481951, "lon":-2.526448}, zoom=13)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

fig.show()
