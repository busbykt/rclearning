'''
This script tests principal component analysis for use by a flight dynamics model.
'''

# %% 
# dependencies
import os
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns

# adjust the working directory
if os.path.basename(os.getcwd()) != 'rclearning':
    os.chdir('..')
# ensure the working directory is rclearning
assert os.path.basename(os.getcwd()) == 'rclearning'

# %%
# read in the dataset
df = pd.read_parquet('./data/firstCrashCleaned.parquet')

# %%
# feature engineering
# change in SSA, AOA, AspdE, Spd
for col in ['SSA','AOA','AspdE','Spd', 'CRt',
            'C1RCOU','C2RCOU','C3RCOU','C4RCOU',
            'GyrX','GyrY','GyrZ',
            'AccX','AccY','AccZ',
            'Roll', 'Pitch']:
    df['d'+col] = df[col].diff()
# %%
# things to predict
yCols = [
    'AspdE', # airspeed error
    'AOA',
    'SSA', # slide slip angle
    'Spd',
    'Roll', 
    'Pitch', 
    'GyrX',
    'GyrY',
    'GyrZ',
    'AccX',
    'AccY',
    'AccZ',
    # 'Alt',
    # 'Press',
    # 'Temp',
    'CRt' # climb rate
]

xCols = [
    'AspdE', # airspeed error
    'AOA',
    'SSA', # slide slip angle
    'Spd',
    'Roll', 
    'Pitch', 
    'GyrX',
    'GyrY',
    'GyrZ',
    'AccX',
    'AccY',
    'AccZ',
    # 'Alt',
    # 'Press',
    # 'Temp',
    'CRt',
    'C1RCOU',
    'C2RCOU',
    'C3RCOU',
    'C4RCOU',
    # engineered cols
    'dC1RCOU',
    'dC2RCOU',
    'dC3RCOU',
    'dC4RCOU',
    'dSSA',
    'dAspdE',
    'dAOA',
    'dSpd',
    'dGyrX',
    'dGyrY',
    'dGyrZ',
    'dAccX',
    'dAccY',
    'dAccZ',
    'dRoll',
    'dPitch',
    'dCRt'
]

# %%
# shift the y samples by 1
y = ['y'+x for x in yCols]
df[y] = df[yCols].shift(1)

# first samples are not usable
df = df.iloc[2:]

# train test split
trainLen = int(len(df)*.7)
xTrain = df[xCols][:trainLen]
xTest = df[xCols][trainLen:]
yTrain = df[y][:trainLen]
yTest = df[y][trainLen:]

# standard scale the dataset to make errors comparable
xScaler = StandardScaler()
xTrain = pd.DataFrame(xScaler.fit_transform(xTrain[xCols]), columns=xCols)
xTest = pd.DataFrame(xScaler.transform(xTest[xCols]), columns=xCols)

yScaler = StandardScaler()
yTrain = pd.DataFrame(yScaler.fit_transform(yTrain[y]), columns=y)
yTest = pd.DataFrame(yScaler.transform(yTest[y]), columns=y)
