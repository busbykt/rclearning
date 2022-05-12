'''
Read a log file into a pandas dataframe.
'''

# %%
# dependencies
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ensure the working directory is rclearning
assert os.path.basename(os.getcwd()) == 'rclearning'

# %%
# read in the csv log files
allFiles = os.listdir("./logs/")    
csvFiles = list(filter(lambda f: f.lower().endswith('.csv'), allFiles))
# %%

for i,file in enumerate(csvFiles):
    
    if i == 0: # if first file, read
        df = pd.read_csv('./logs/'+file)
    else: # otherwise append it to the dataframe
        df = df.merge(pd.read_csv('./logs/'+file),
                      how='outer',
                      left_on='timestamp',
                      right_on='timestamp',
                      suffixes=(None,file[-8:-4]))

# sort the data by timestamp
df.sort_values('timestamp', inplace=True)
# forward fill data where missing
df.interpolate(method='linear', inplace=True, limit_direction='forward')

# drop the first missing records
df = df[~df.isnull().max(axis=1)]
# reset the index
df.reset_index(inplace=True, drop=True)

# make the timestamp a datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# make datetime the index
df.index = df['datetime']

# add a feature to track the timestep as the data is async
df['timeStep'] = df['datetime'].shift(-1) - df['datetime']

# resample the data at some fixed frequency
df.resample('50ms').ffill()

# only keep records when the airplane is likely to be flying
df = df[df['isFlyProb'] > .8]

df['mode'] = np.where(df['C8'] < 1000, 'manual', 'fbwa')
df['mode'] = np.where(df['C8'] > 2000, 'autotune', df['mode'])

# %%
fig,ax = plt.subplots(dpi=100, figsize=[6,5])
norm = plt.Normalize(df['timestamp'].min(), df['timestamp'].max())
sm = plt.cm.ScalarMappable(cmap="Spectral", norm=norm)
sm.set_array([])
sns.scatterplot(
    data=df[(df['mode'] == 'manual')], 
    x='C2RCOU', 
    y='GyrY', 
    hue='datetime',
    legend=False,
    ax=ax,
    palette='Spectral'
    )
ax.figure.colorbar(sm)
# %%
# scale the data so it may be plotted together cleanly
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# only scale numeric cols
colsToScale = df.select_dtypes('float').columns.tolist()

# scale the data and save into another dataframe
scaledDF = pd.DataFrame(scaler.fit_transform(df[colsToScale]), 
                        columns=colsToScale)
scaledDF['mode'] = df['mode'].values
# %%
fig,ax = plt.subplots(dpi=90, figsize=[8,6])
plt.plot(scaledDF[scaledDF['mode'].isin(['fbwa', 'manual'])]['AccX'])
plt.plot(scaledDF[scaledDF['mode'].isin(['fbwa', 'manual'])]['Roll'])
plt.xlim(0,16000)