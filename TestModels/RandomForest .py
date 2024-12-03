import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/Avina/OneDrive/Documents/GitHub/Power-Delhi/megaweadata.csv')

# Convert the 'Date' column to datetime and set it as the index
df['TIMESLOT'] = pd.to_datetime(df['TIMESLOT'])
df.set_index('TIMESLOT', inplace=True)


#print(df.shape)
#print(df.info())
duplicate_rows_data = df[df.duplicated()]
#print("number of duplicate rows: ", duplicate_rows_data.shape)

df.drop_duplicates(inplace=True)


# missing value query
df.dropna(inplace=True)


num_list = ['DELHI','dayofWeek','temp','feelslike','dew','humidity','windspeed','isHoliday']

num_var = pd.Series(int(x) for x in num_list['DELHI'], name = "Numerical Variable")
 
# Plot histogram
sns.histplot(data = num_var, kde = True)