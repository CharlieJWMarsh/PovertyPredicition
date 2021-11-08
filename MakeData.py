import sqlite3
import pandas as pd
import csv


pd.set_option('display.max_columns', 100)  # or 1000
pd.set_option('display.max_rows', 100)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

KS101EW = pd.read_csv(r'C:\Users\charl\OneDrive\Documents\2011 Census\KS101EW.csv')
KS102EW = pd.read_csv(r'C:\Users\charl\OneDrive\Documents\2011 Census\KS102EW.csv')

# for col in KS102EW.columns:
#     print(col)

conn = sqlite3.connect('Census_2011_database.db')
c = conn.cursor()

KS101EW.to_sql("all_data", conn, if_exists='append', index=False)
KS101EW.to_sql("KS101EW", conn, if_exists='append', index=False)
KS102EW.to_sql("KS102EW", conn, if_exists='append', index=False)

c.execute("""
SELECT *
FROM all_data
INNER JOIN KS102EW
ON all_data.geography = KS102EW.geography
""")
print(c.fetchall())

# print(pd.read_sql_query("SELECT * FROM all_data", conn))


conn.commit()
conn.close()
