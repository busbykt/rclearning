'''
This script builds a data-driven flight dynamics model using a recurrent neural network.
A recurrent neural network may be useful in lieu of adding derivative or integral components
of features
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
from keras.layers import Dense, SimpleRNN

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
    'C4RCOU'
]

# %%
# shift the y samples by 1
y = ['y'+x for x in yCols]
df[y] = df[yCols].shift(-1)

# last and first samples are not usable
df = df.iloc[1:-1]

# train test split
trainLen = int(len(df)*.7)
xTrain = df[xCols][:trainLen]
xTest = df[xCols][trainLen:]
yTrain = df[y][:trainLen]
yTest = df[y][trainLen:]

yTestUnscaled = yTest.copy()
xTestUnscaled = xTest.copy()

# standard scale the dataset to make errors comparable
xScaler = StandardScaler()
xTrain = pd.DataFrame(xScaler.fit_transform(xTrain[xCols]), columns=xCols)
xTest = pd.DataFrame(xScaler.transform(xTest[xCols]), columns=xCols)

yScaler = StandardScaler()
yTrain = pd.DataFrame(yScaler.fit_transform(yTrain[y]), columns=y)
yTest = pd.DataFrame(yScaler.transform(yTest[y]), columns=y)

# %%
# create a dataset of sliding windows of inputs and targets
sequenceLen = 3
samplingRate = 1
batchSize=16
datasetTrain = keras.preprocessing.timeseries_dataset_from_array(
    xTrain.values,
    yTrain.values,
    sequence_length=sequenceLen,
    sampling_rate=samplingRate,
    batch_size=batchSize,
    end_index=int(len(xTrain)*.75)
)
# %%
# create validation timeseries dataset
datasetVal = keras.preprocessing.timeseries_dataset_from_array(
    xTrain.values,
    yTrain.values,
    sequence_length=sequenceLen,
    sampling_rate=samplingRate,
    batch_size=batchSize,
    start_index=int(len(xTrain)*.7)
)
# create test timeseries dataset
datasetTest = keras.preprocessing.timeseries_dataset_from_array(
    xTest,
    yTest,
    sequence_length=sequenceLen,
    sampling_rate=samplingRate,
    batch_size=batchSize
)

# %%
for batch in datasetTrain.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

# %%

model = keras.Sequential()
model.add(SimpleRNN(64,activation='relu'))
model.add(Dense(units=targets.numpy().shape[1], activation='relu'))
model.compile(loss='mse', optimizer='adam')
# %%
history = model.fit(datasetTrain, validation_data=datasetVal, epochs=25)
# %%
# plot train and validation results
modelHist = pd.DataFrame(history.history)
fig,ax = plt.subplots()
sns.lineplot(data=modelHist,ax=ax)
# ax.set_yscale('log')
plt.legend()
plt.show()
# %%
yTestPred = pd.DataFrame(yScaler.inverse_transform(model.predict(datasetTest)), columns=yCols)
# %%
