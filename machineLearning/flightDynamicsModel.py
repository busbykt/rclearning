'''
This script builds a data-driven flight dynamics model.
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
for col in ['SSA','AOA','AspdE','Spd',
            'C1RCOU','C2RCOU','C3RCOU','C4RCOU']:
    df['d'+col] = df[col].diff()
# %%
# things to predict
yCols = [
    'AspdE', # airspeed error (negative??)
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
    'AccZ'
]

xCols = [
    'AspdE', # airspeed error (negative??)
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
    'dSpd'
]

# %%
# shift the y samples by 1
y = ['y'+x for x in yCols]
df[y] = df[yCols].shift(1)

# first sample is not usable
df = df.iloc[1:]

# standard scale the dataset to make errors comparable
scaler = StandardScaler()
dfs = pd.DataFrame(scaler.fit_transform(df[xCols+y]), columns=xCols+y)

# train test split
trainLen = int(len(dfs)*.75)
xTrain = dfs[xCols][:trainLen]
xTest = dfs[xCols][trainLen:]
yTrain = dfs[y][:trainLen]
yTest = dfs[y][trainLen:]
# %%
# build up a NN model
model = keras.Sequential()
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dense(64))
model.add(keras.layers.Dense(12))

# %%
model.compile(optimizer='sgd', loss='mse')
model.fit(xTrain,yTrain, batch_size=16, epochs=25, validation_split=.2)

# %%
dfPred = pd.DataFrame(model.predict(xTest), columns=y)

# %%
# build up a rates model
def ratePred(data, xCols, yCols):
    '''
    Makes a prediction about the next state of the aircraft given
    its current state incorporating rates.
    '''
    # copy the dataframe
    df = data.copy()

    # compute delta for each row
    for col in xCols:
        df['d'+col] = df[col].diff()

    for col in yCols:
        if col in xCols:
            df['y'+col] = df[col]+df['d'+col]
        else:
            df['y'+col] = df[col]

    df.fillna(method='bfill', inplace=True)

    return pd.DataFrame(df[['y'+x for x in yCols]].to_numpy(), columns=yCols)

ratePreds = ratePred(xTest,xCols,yCols)

# %%
for col in y:
    plt.scatter(dfPred[col],yTest[col], label='NN', alpha=.7)
    plt.scatter(xTest[col[1:]], yTest[col], label='dummy', alpha=.7)
    # plt.scatter(ratePreds[col[1:]], yTest[col], label='rate', alpha=.7)
    plt.title(col)
    plt.legend()
    plt.xlabel('pred'); plt.ylabel('truth')
    plt.plot([dfPred[col].min(),dfPred[col].max()],
             [dfPred[col].min(),dfPred[col].max()], c='k', ls='--')
    plt.show()
    print(col)
    print(np.round(mean_squared_error(yTest[col],dfPred[col]),4),'NN')
    print(np.round(mean_squared_error(yTest[col],xTest[col[1:]]),4),'dummy')
    print(np.round(mean_squared_error(yTest[col],ratePreds[col[1:]]),4),'rate')
# %%

print('dummyModel', np.round(mean_squared_error(yTest,xTest[[x[1:] for x in y]]),3))
print('NN',np.round(mean_squared_error(yTest,dfPred),3))
# %%
