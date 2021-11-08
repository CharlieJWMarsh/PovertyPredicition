import sqlite3
import pandas as pd
import csv


KS101EW = pd.read_csv(r'C:\Users\charl\OneDrive\Documents\2011 Census\KS101EW.csv')
KS102EW = pd.read_csv(r'C:\Users\charl\OneDrive\Documents\2011 Census\KS102EW.csv')

# for col in KS102EW.columns:
#     print(col)

sql_connect = sqlite3.connect('Census_2011_database.db')
c = sql_connect.cursor()

KS101EW.to_sql("KS101EW", sql_connect, if_exists='append', index=False)
KS102EW.to_sql("KS102EW", sql_connect, if_exists='append', index=False)

c.execute("SELECT * FROM KS102EW")
print(c.fetchall())

# c.execute('''
# CREATE TABLE KS101EW(
# date INTEGER,
# geography VARCHAR,
# geography_code VARCHAR
# rural_urban VARCHAR,
# All_usual_residents INTEGER,
# Males INTEGER,
# Females INTEGER,
# Lives_in_a_household INTEGER,
# Lives_in_a_communal_establishment INTEGER,
# Student VARCHAR,
# Area_Hectares DOUBLE,
# Density_per_Hectare DOUBLE);
# ''')
#
# contents = csv.reader(open(r'C:\Users\charl\OneDrive\Documents\2011 Census\KS101EW.csv'))
#
# c.executemany("INSERT INTO KS101EW VALUES", contents)

sql_connect.commit()
sql_connect.close()

