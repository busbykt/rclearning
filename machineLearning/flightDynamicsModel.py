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
    'AccZ',
    'Alt',
    'Press',
    # 'Temp',
    'CRt'
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
    'Alt',
    'Press',
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
yScaler = StandardScaler()
yScaler.fit(df[y])

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
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dense(64))
model.add(keras.layers.Dense(len(yCols)))

# %%
model.compile(optimizer='sgd', loss='mse')
history = model.fit(xTrain,yTrain, batch_size=32, epochs=30, validation_split=.2)

# %%
# plot train and validation results
modelHist = pd.DataFrame(history.history)
fig,ax = plt.subplots()
sns.lineplot(data=modelHist,ax=ax)
ax.set_xscale('log')
plt.legend()
plt.show()
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
# convert back to original units for plotting

# %%
for col in y:
    fig,ax = plt.subplots(dpi=80, figsize=[7,5])
    ax.scatter(dfPred[col],yTest[col], label='NN', alpha=.7)
    ax.scatter(xTest[col[1:]], yTest[col], label='dummy', alpha=.7)
    # ax.scatter(ratePreds[col[1:]], yTest[col], label='rate', alpha=.7)
    ax.set_title(col)
    ax.set_xlabel('pred'); ax.set_ylabel('truth')
    ax.plot([dfPred[col].min(),dfPred[col].max()],
             [dfPred[col].min(),dfPred[col].max()], 
             c='k', ls='--', label='perfect model')
    ax.legend()
    plt.show()
    print(col)
    print(np.round(mean_squared_error(yTest[col],dfPred[col]),4),'NN')
    print(np.round(mean_squared_error(yTest[col],xTest[col[1:]]),4),'dummy')
    print(np.round(mean_squared_error(yTest[col],ratePreds[col[1:]]),4),'rate')
# %%
print('dummyModel', np.round(mean_squared_error(yTest,xTest[[x[1:] for x in y]]),3))
print('rateModel', np.round(mean_squared_error(yTest,ratePreds[[x[1:] for x in y]]),3))
print('NN',np.round(mean_squared_error(yTest,dfPred),3))
# %%
# error distributions
yTestUnscaled = pd.DataFrame(yScaler.transform(yTest), columns=yCols)
dfPredUnscaled = pd.DataFrame(yScaler.transform(dfPred), columns=yCols)
errorDF = yTestUnscaled - dfPredUnscaled
# %%
sns.histplot(errorDF[['AccX','AccY','AccZ']])
plt.xlabel('Acceleration m/s2');
# %%
sns.histplot(errorDF[['GyrX','GyrY','GyrZ']])
plt.xlabel('rad/s');
# %%
sns.histplot(errorDF[['Roll','SSA','AOA','Pitch']])
plt.xlabel('degrees');
# %%
sns.histplot(errorDF[['AspdE','Spd']])
# %%
