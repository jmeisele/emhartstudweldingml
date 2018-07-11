#Import libraries
import glob
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(color_codes = True)

#Create a list of all csv files
path = "C:\\Users\\EISELJA\\Desktop\\EmhartData\\ProcessData\\*.csv"
allFiles = glob.glob(path)

#Create a DataFrame of all csv files
df = pd.concat((pd.read_csv(f) for f in allFiles), ignore_index = True, axis =0)

# Drop rows with all NaNs
df = df.dropna(how = 'all')

#Drop Columns with which contain only reference or static values
df= df.drop(columns = ['Peak / Drop voltage','Remark','Pilot Weldcurrent Arc Voltage  Reference (Up)',\
                       'Main Weldcurrent Voltage  Reference (Us)','Weldcurrent Minimum (Is)',\
                       'Weldcurrent Maximum (Is)','Weld Energy  Reference (Es)',\
                       'Weld Energy  Minimum (Es)','Weld Energy  Maximum (Es)','Carbody ID:'])

#Start cleaning and transforming the data
df['Application'] = df['Application'].astype('category')
df['Date / Time'] = df['Date / Time'].astype(np.datetime64)
df['Device Name'] = df['Device Name'].astype('category')
df['Type'] = df['Type'].astype('category')
df['Outlet/Feeder/Tool'] = df['Outlet/Feeder/Tool'].astype('category')
df['Tool type'] = df['Tool type'].astype('category')  
df['MAC process'] = df['MAC process'].astype('category')
df['Fault number'] = df['Fault number'].replace('-', np.NaN)
df['Fault number'] = pd.to_numeric(df['Fault number'])
df['Mode'] = df['Mode'].replace('-', np.NaN)
df['Mode'] = df['Mode'].astype('category')
df['Message:'] = df['Message:'].replace('-', np.NaN)
df['Message:'] = df['Message:'].astype('category')
df['Stud-ID:'] = df['Stud-ID:'].astype('category')
df['Optimization'] = df['Optimization'].astype('category')
df['Clean time'] = df['Clean time'].str.extract('(\d+)', expand=True).astype(int)
df['Weld mode'] = df['Weld mode'].astype('category')
df['Pilot Weldcurrent Arc Voltage  Minimum (Up)'] = df['Pilot Weldcurrent Arc Voltage  Minimum (Up)'].replace('-', np.NaN)
df['Pilot Weldcurrent Arc Voltage  Minimum (Up)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Pilot Weldcurrent Arc Voltage  Minimum (Up)'] = pd.to_numeric(df['Pilot Weldcurrent Arc Voltage  Minimum (Up)'])
df['Pilot Weldcurrent Arc Voltage  Maximum (Up)'] = df['Pilot Weldcurrent Arc Voltage  Maximum (Up)'].replace('-', np.NaN)
df['Pilot Weldcurrent Arc Voltage  Maximum (Up)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Pilot Weldcurrent Arc Voltage  Maximum (Up)'] = pd.to_numeric(df['Pilot Weldcurrent Arc Voltage  Maximum (Up)'])
df['Pilot Weldcurrent Arc Voltage  Actual (Up)'] = df['Pilot Weldcurrent Arc Voltage  Actual (Up)'].str.extract('(\d+)', expand=True).astype(int)
df['Main Weldcurrent Voltage  Minimum (Us)'] = df['Main Weldcurrent Voltage  Minimum (Us)'].str.extract('(\d+)', expand=True).astype(int)
df['Main Weldcurrent Voltage  Maximum (Us)'] = df['Main Weldcurrent Voltage  Maximum (Us)'].str.extract('(\d+)', expand = True).astype(int)
df['Main Weldcurrent Voltage  Actual (Us)'] = df['Main Weldcurrent Voltage  Actual (Us)'].str.extract('(\d+)', expand = True).astype(int)
df['Weldtime  Reference (It)'] = df['Weldtime  Reference (It)'].str.extract('(\d+)', expand = True).astype(int)
df['Weldtime  Reference (It)'] = pd.to_numeric(df['Weldtime  Reference (It)'])
df['Weldtime  Minimum (It)'] = df['Weldtime  Minimum (It)'].replace('-', np.NaN)
df['Weldtime  Minimum (It)'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
df['Weldtime  Minimum (It)'] = pd.to_numeric(df['Weldtime  Minimum (It)'])
df['Weldtime  Reference (It)'] = pd.to_numeric(df['Weldtime  Reference (It)'])
df['Weldtime  Maximum (It)'] = df['Weldtime  Maximum (It)'].replace('-', np.NaN)
df['Weldtime  Maximum (It)'].replace(regex = True, inplace = True, to_replace =r'\D', value = r'')
df['Weldtime  Maximum (It)'] = pd.to_numeric(df['Weldtime  Maximum (It)'])
df['Weldtime  Actual (It)'].replace(regex = True, inplace = True, to_replace =r'\D', value = r'')
df['Weldtime  Actual (It)'] = pd.to_numeric(df['Weldtime  Actual (It)'])
df['Weld Energy  Actual (Es)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Weld Energy  Actual (Es)'] = pd.to_numeric(df['Weld Energy  Actual (Es)'])
df['Lift height Reference'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Lift height Reference'] = pd.to_numeric(df['Lift height Reference'])
df['Lift height Minimum'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Lift height Minimum'] = pd.to_numeric(df['Lift height Minimum'])
df['Lift height Maximum'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Lift height Maximum'] = pd.to_numeric(df['Lift height Maximum'])
df['Lift height Actual'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Lift height Actual'] = pd.to_numeric(df['Lift height Actual'])
df['Penetration  Reference (P)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Penetration  Reference (P)'] = pd.to_numeric(df['Penetration  Reference (P)'])
df['Penetration  Minimum (P)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Penetration  Minimum (P)'] = pd.to_numeric(df['Penetration  Minimum (P)'])
df['Penetration  Maximum (P)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Penetration  Maximum (P)'] = pd.to_numeric(df['Penetration  Maximum (P)'])
df['Penetration  Actual (P)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Penetration  Actual (P)'] = pd.to_numeric(df['Penetration  Actual (P)'])

#Split duplicate values into seperate columns and strip the alpha characters out
df['Weldcurrent Reference (A)'], df['Weldcurrent Reference (B)'] = df['Weldcurrent Reference (Is)'].str.split('/', 1).str
df['Weldcurrent Reference (A)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Weldcurrent Reference (A)'] = pd.to_numeric(df['Weldcurrent Reference (A)'])
df['Weldcurrent Reference (B)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Weldcurrent Reference (B)'] = pd.to_numeric(df['Weldcurrent Reference (B)'])
df['Weldcurrent Actual (A)'], df['Weldcurrent Actual (B)'] = df['Weldcurrent Actual (Is)'].str.split('/', 1).str
df['Weldcurrent Actual (A)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Weldcurrent Actual (A)'] = pd.to_numeric(df['Weldcurrent Actual (A)'])
df['Weldcurrent Actual (B)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['Weldcurrent Actual (B)'] = pd.to_numeric(df['Weldcurrent Actual (B)'])
df['DroptimeRef(ms)'], df['DroptimeActual(ms)'] = df['Droptime (Ref / Actual)'].str.split('/', 1).str
df['DroptimeRef(ms)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['DroptimeRef(ms)'] = pd.to_numeric(df['DroptimeRef(ms)'])
df['DroptimeActual(ms)'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['DroptimeActual(ms)'] = pd.to_numeric(df['DroptimeActual(ms)'])
df['StickoutRef'], df['StickoutActual'] = df['Stickout (Ref / Actual)'].str.split('/', 1).str
df['StickoutRef'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['StickoutRef'] = pd.to_numeric(df['StickoutRef'])
df['StickoutActual'].replace(regex = True, inplace = True, to_replace = r'\D', value = r'')
df['StickoutActual'] = pd.to_numeric(df['StickoutActual'])
df['Outlet'], df['Feeder'], df['Tool'] = df['Outlet/Feeder/Tool'].str.split('/', 2).str
df['Outlet'] = pd.to_numeric(df['Outlet'])
df['Feeder'] = pd.to_numeric(df['Feeder'])
df['Tool'] = pd.to_numeric(df['Tool'])

# Drop records where Lift height Actual contains NaN values
df.dropna(subset = ['Lift height Actual'], inplace = True)

# Drop unused columns
df= df.drop(columns = ['Weldcurrent Reference (Is)','Stickout (Ref / Actual)',\
                       'Droptime (Ref / Actual)','Weldcurrent Actual (Is)','Stickout (Ref / Actual)',\
                       'Outlet/Feeder/Tool','StickoutRef'])

# Seperate features which are actuals from those that are static reference lines or static maximum/minimum values
df = df[['Type', 'Device Name', 'Fault number', 'Stud-ID:', 'Date / Time',
       'Application', 'Tool type','System weld  counter', 'Tool weld  counter', 'Outlet weld counter WOP',
       'Outlet weld counter', 'Optimization', 'Clean time', 'Mode',
       'Weld mode','Pilot Weldcurrent Arc Voltage  Actual (Up)','Main Weldcurrent Voltage  Actual (Us)', 
                   'Weldtime  Actual (It)','Weld Energy  Actual (Es)','Lift height Actual','Penetration  Actual (P)',
                  'Weldcurrent Actual (A)','Weldcurrent Actual (B)','DroptimeActual(ms)','StickoutActual','Outlet', 'Feeder', 'Tool']]

#Drop categorical columns
df.drop(columns = ['Fault number','Stud-ID:', 'Date / Time','Application', 'Tool type',
                   'System weld  counter','Tool weld  counter', 'Outlet weld counter WOP','Outlet weld counter',
                  'Optimization', 'Mode', 'Weld mode', 'Outlet','Feeder', 'Tool'],inplace = True)

#Move the response variable to the front
front = df['Type']
df.drop(columns = ['Type'], axis = 1, inplace = True)
df.insert(0, 'Type', front)
df.head()

# TODO # Generate csv of tidy data
#df.to_csv("C:\\Users\\EISELJA\\Desktop\\EmhartData\\ProcessData.csv")

#Sample only FS73 160RB400
rb400 = df[df['Device Name'] == 'FS73 160RB400']
rb400.drop(columns = 'Device Name', inplace = True, axis = 1)


# Aggregate the dataset between independent variables X and dependent variable y
X = rb400.iloc[:, 1:12]
y = rb400.iloc[:, 0]

# Encode categorical data [Device Name]
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
#X.iloc[:, 0] = enc.fit_transform(X.iloc[:, 0])
y = enc.fit_transform(y)

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Normalize the data with feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
