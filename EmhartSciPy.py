#Import libraries
import os
import csv
import glob
import numpy as np
import pandas as pd

#Create a list of all csv files
path = "C:\\Users\\EISELJA\\Desktop\\EmhartData\\ProcessData\\*.csv"
allFiles = glob.glob(path)

#Create a DataFrame of all csv files
df = pd.concat((pd.read_csv(f) for f in allFiles), ignore_index = True, axis =0)

# Drop rows with all NaNs
df = df.dropna(how = 'all')

#Drop Columns with NaNs
df= df.drop(columns = ['Peak / Drop voltage','Remark','Pilot Weldcurrent Arc Voltage  Reference (Up)',\
                       'Main Weldcurrent Voltage  Reference (Us)','Weldcurrent Minimum (Is)',\
                       'Weldcurrent Maximum (Is)','Weld Energy  Reference (Es)',\
                       'Weld Energy  Minimum (Es)','Weld Energy  Maximum (Es)'])

#Start cleaning and transforming the data
df['Application'] = df['Application'].astype('category')
df['Date / Time'] = df['Date / Time'].astype(np.datetime64)
df['Device Name'] = df['Device Name'].astype('category')
df['Type'] = df['Type'].astype('category')
df['Outlet/Feeder/Tool'] = df['Outlet/Feeder/Tool'].astype('category')
df['Tool type'] = df['Tool type'].astype('category')  
df['MAC process'] = df['MAC process'].astype('category')
df['Fault number'] = df['Fault number'].replace('-', np.NaN)
df['Mode'] = df['Mode'].replace('-', np.NaN)
df['Mode'] = df['Mode'].astype('category')
df['Message:'] = df['Message:'].replace('-', np.NaN)
df['Message:'] = df['Message:'].astype('category')
df['Stud-ID:'] = df['Stud-ID:'].astype('category')
df['Optimization'] = df['Optimization'].astype('category')
df['Clean time'] = df['Clean time'].str.extract('(\d+)', expand=True).astype(int)
df['Weld mode'] = df['Weld mode'].astype('category')
df['Pilot Weldcurrent Arc Voltage  Minimum (Up)'] = df['Pilot Weldcurrent Arc Voltage  Minimum (Up)'].replace('-', np.NaN)
df['Pilot Weldcurrent Arc Voltage  Maximum (Up)'] = df['Pilot Weldcurrent Arc Voltage  Maximum (Up)'].replace('-', np.NaN)
df['Pilot Weldcurrent Arc Voltage  Actual (Up)'] = df['Pilot Weldcurrent Arc Voltage  Actual (Up)'].str.extract('(\d+)', expand=True).astype(int)
df['Main Weldcurrent Voltage  Minimum (Us)'] = df['Main Weldcurrent Voltage  Minimum (Us)'].str.extract('(\d+)', expand=True).astype(int)
df['Main Weldcurrent Voltage  Maximum (Us)'] = df['Main Weldcurrent Voltage  Maximum (Us)'].str.extract('(\d+)', expand = True).astype(int)
df['Main Weldcurrent Voltage  Actual (Us)'] = df['Main Weldcurrent Voltage  Actual (Us)'].str.extract('(\d+)', expand = True).astype(int)
df['Weldtime  Reference (It)'] = df['Weldtime  Reference (It)'].str.extract('(\d+)', expand = True).astype(int)
df['Weldtime  Minimum (It)'] = df['Weldtime  Minimum (It)'].replace('-', np.NaN)
df['Weldtime  Minimum (It)'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
df['Weldtime  Maximum (It)'] = df['Weldtime  Maximum (It)'].replace('-', np.NaN)
df['Weldtime  Maximum (It)'].replace(regex = True, inplace = True, to_replace =r'\D', value = r'')
df['Weldtime  Actual (It)'].replace(regex = True, inplace = True, to_replace =r'\D', value = r'')
df['Weld Energy  Actual (Es)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Lift height Reference'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Lift height Minimum'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Lift height Maximum'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Lift height Actual'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Penetration  Reference (P)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Penetration  Minimum (P)'].replace(regex = True, inplace = True, to_replace = r'\D', value = '')
df['Penetration  Maximum (P)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Penetration  Actual (P)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')

#Split duplicate values into seperate columns
df['Weldcurrent Reference (Is)']
df['Weldcurrent Actual (Is)']

# Strip the white space from cells and replace with 0's
df['Carbody ID:'] = df['Carbody ID:'].str.strip()
df['Carbody ID:'] = df['Carbody ID:'].fillna(0)

# We cant convert since there are categorical values such as 'Ghost' and 'Hybrid'
#df['Carbody ID:'] = df['Carbody ID:'].apply(pd.to_numeric)

# Generate csv of tidy data
df.to_csv("C:\\Users\\EISELJA\\Desktop\\EmhartData\\ProcessData\\Test.csv")
