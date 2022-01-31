import pandas as pd
import json

bristol_json = json.load(open('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\Bristol_OA.json', 'r'))

geo_df = pd.read_csv('C:\\Users\\charl\\OneDrive\\Documents\\2011 Census\\BristolheatmapData.csv')
OA = geo_df['OA'][0]

print(OA)

print(bristol_json["id"])
