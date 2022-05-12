'''
This script builds a data-driven flight dynamics model.
'''

# %% 
# dependencies
import os
from tensorflow import keras
import pandas as pd
import numpy as np
import scipy as sp

# adjust the working directory
if os.path.basename(os.getcwd()) != 'rclearning':
    os.chdir('..')
# ensure the working directory is rclearning
assert os.path.basename(os.getcwd()) == 'rclearning'

# %%
# read in the dataset
df = pd.read_parquet('./data/firstCrashCleaned.parquet')

# dont use the crash...

# things to predict
yCols = [
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
    'Roll', 
    'Pitch', 
    'GyrX',
    'GyrY',
    'GyrZ',
    'AccX',
    'AccY',
    'AccZ',
    'RCOU1',
    'RCOU2',
    'RCOU3',
    'RCOU4'
]

# %%
# split the data into train test
# %%
# build up a NN model
model = keras.Sequential()
model.add(keras.layers.Dense(64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dense(8))

# %%
model.compile(optimizer='sgd', loss='mse')
model.fit(xTrain,yTrain, batch_size=512, epochs=10)