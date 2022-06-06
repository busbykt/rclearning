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
df[y] = df[yCols].shift(-1)

# last and first samples are not usable
df = df.iloc[1:-1]

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

# %%
# build up a NN model
model = keras.Sequential()
model.add(keras.layers.Dense(1024, activation='relu'))
# randomly set input units to 0 during training
model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Dense(len(yCols)))

# %%
model.compile(optimizer='sgd', loss='mse')
history = model.fit(xTrain,yTrain, batch_size=16, epochs=150, validation_split=.3)

# %%
# plot train and validation results
modelHist = pd.DataFrame(history.history)
fig,ax = plt.subplots()
sns.lineplot(data=modelHist,ax=ax)
# ax.set_yscale('log')
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
print('dummyModel', np.round(mean_squared_error(yTest,xTest[[x[1:] for x in y]]),3))
print('rateModel', np.round(mean_squared_error(yTest,ratePreds[[x[1:] for x in y]]),3))
print('NN',np.round(mean_squared_error(yTest,dfPred),3))
# %%
xTrainUnscaled = pd.DataFrame(xScaler.inverse_transform(xTrain), columns=xCols)
yTestUnscaled = pd.DataFrame(yScaler.inverse_transform(yTest), columns=y)
dfPredUnscaled = pd.DataFrame(yScaler.inverse_transform(dfPred), columns=y)
xTestUnscaled = pd.DataFrame(xScaler.inverse_transform(xTest), columns=xCols)
ratePredsUnscaled =  pd.DataFrame(yScaler.inverse_transform(
    ratePreds[[x[1:] for x in y]]), columns=y)

for col in y:
    fig,ax = plt.subplots(dpi=80, figsize=[7,5])
    ax.scatter(dfPredUnscaled[col],yTestUnscaled[col], label='NN', alpha=.7)
    ax.scatter(xTestUnscaled[col[1:]], yTestUnscaled[col], label='dummy', alpha=.7)
    # ax.scatter(ratePredsUnscaled[col], yTestUnscaled[col], label='rate', alpha=.7)
    ax.set_title(col)
    ax.set_xlabel('pred'); ax.set_ylabel('truth')
    ax.plot([dfPredUnscaled[col].min(),dfPredUnscaled[col].max()],
             [dfPredUnscaled[col].min(),dfPredUnscaled[col].max()], 
             c='k', ls='--', label='perfect model')
    ax.legend()
    plt.show()
    print(col)
    print(np.round(mean_squared_error(yTest[col],dfPred[col]),4),'NN')
    print(np.round(mean_squared_error(yTest[col],xTest[col[1:]]),4),'dummy')
    print(np.round(mean_squared_error(yTest[col],ratePreds[col[1:]]),4),'rate')
# %%
# error distributions
errorDF = yTestUnscaled - dfPredUnscaled
# %%
sns.histplot(errorDF[['yAccX','yAccY','yAccZ']])
plt.xlabel('Acceleration m/s2');
# %%
sns.histplot(errorDF[['yGyrX','yGyrY','yGyrZ']])
plt.xlabel('rad/s');
# %%
sns.histplot(errorDF[['yRoll','ySSA','yAOA','yPitch']])
plt.xlabel('degrees');
# %%
sns.histplot(errorDF[['yAspdE','ySpd']])

# %%
# interpret how the model responds to changes in control inputs
varDefDict = {'C1RCOU':'Aileron Position', 
              'C2RCOU':'Elevator Position', 
              'C3RCOU':'Throttle Position', 
              'C4RCOU':'Rudder Position'}
for controlInput in varDefDict.keys():
    samples = int(len(xTest)/2)
    steps = 100
    xTestICE = pd.DataFrame(np.repeat(xTestUnscaled.sample(samples).values,steps,axis=0), 
                                    columns=xTest.columns)
    inputs = [np.linspace(xTrainUnscaled[controlInput].min(),
                          xTrainUnscaled[controlInput].max(),
                          steps) for _ in range(samples)]
    xTestICE[controlInput] = np.stack(inputs, axis=0).flatten()
    xTestICE = pd.DataFrame(xScaler.transform(xTestICE),columns=xCols)
    ICEpred = pd.DataFrame(model.predict(xTestICE), columns=yCols)
    # unscale the inputs and predictions
    xTestICEUnscaled = pd.DataFrame(xScaler.inverse_transform(xTestICE),columns=xCols)
    ICEpredUnscaled = pd.DataFrame(yScaler.inverse_transform(ICEpred), columns=yCols)

    # add an index to groupby
    xTestICEUnscaled['_index'] = np.stack([range(steps)]*samples,axis=0).flatten()
    ICEpredUnscaled['_index'] = np.stack([range(steps)]*samples,axis=0).flatten()

    # groupby the new index taking the mean
    xTestPDPUnscaled = xTestICEUnscaled.groupby('_index').mean()
    PDPpredUnscaled = ICEpredUnscaled.groupby('_index').mean()

    # convert rotation rates to degrees per second
    PDPpredUnscaled[['GyrX','GyrY','GyrZ']] = PDPpredUnscaled[['GyrX','GyrY','GyrZ']] * 180/np.pi 

    # plot the partial dependence of rates
    fig,ax = plt.subplots(figsize=[7,5],dpi=100)
    ax.scatter(xTestPDPUnscaled[controlInput], 
            PDPpredUnscaled['GyrX']-PDPpredUnscaled['GyrX'].mean(), 
            s=5, 
            label='Roll Rate')
    ax.scatter(xTestPDPUnscaled[controlInput], 
            PDPpredUnscaled['GyrY']-PDPpredUnscaled['GyrY'].mean(), 
            s=5, 
            label='Pitch Rate')
    ax.scatter(xTestPDPUnscaled[controlInput], 
            PDPpredUnscaled['GyrZ']-PDPpredUnscaled['GyrZ'].mean(), 
            s=5, 
            label='Yaw Rate')
    ax.axhline(0, ls='--', c='k')
    ax.legend();
    ax.set_xlabel(f'{varDefDict[controlInput]}')
    ax.set_ylabel('Degrees per Second')
    ax.set_title(f'Test Set Partial Dependence on {varDefDict[controlInput]}')
    ax.set_ylim(-45,45)
    ax.grid(alpha=.15)
    plt.show()

    # plot the partial dependence of roll,pitch,ssa
    if controlInput == 'C3RCOU':
        fig,ax = plt.subplots(figsize=[7,5],dpi=100)
        ax.scatter(xTestPDPUnscaled[controlInput], 
                PDPpredUnscaled['AccX']-PDPpredUnscaled['AccX'].mean(), 
                s=5, 
                label='Longitudinal Acceleration m/s^2')
        ax.scatter(xTestPDPUnscaled[controlInput], 
                PDPpredUnscaled['AccY']-PDPpredUnscaled['AccY'].mean(), 
                s=5, 
                label='Lateral Acceleration m/s^2')
        ax.scatter(xTestPDPUnscaled[controlInput], 
                PDPpredUnscaled['AccZ']-PDPpredUnscaled['AccZ'].mean(), 
                s=5, 
                label='Vertical Axis Acceleration m/s^2')
        ax.legend();
        ax.set_xlabel(f'{varDefDict[controlInput]}')
        ax.set_ylabel('meters per second')
        ax.set_title(f'Test Set Partial Dependence on {varDefDict[controlInput]}')
        ax.set_ylim(-1,1)
        ax.axhline(0, ls='--', c='k')
        ax.grid(alpha=.15)
        plt.show()
# %%
controlInput = 'C2RCOU'
a = xTest.sample(1, random_state=3)
interestCols = ['AOA', 'SSA', 'Spd', 'Roll', 'Pitch', 'GyrX', 'C1RCOU']
a = pd.DataFrame(a.values.repeat(10, axis=0), columns=xCols)
a[controlInput] = np.linspace(xTest[controlInput].min(),xTest[controlInput].max(),10)
aUnscaled = pd.DataFrame(xScaler.inverse_transform(a), columns=xCols)

aPred = pd.DataFrame(model.predict(a), columns=y)
aPredUnscaled = pd.DataFrame(yScaler.inverse_transform(aPred), columns=y)
aPredUnscaled[['yGyrX','yGyrY','yGyrZ']] = aPredUnscaled[['yGyrX','yGyrY','yGyrZ']] * 180/np.pi 

# %%
fig,ax = plt.subplots(dpi=100)
ax.scatter(aUnscaled[controlInput],aPredUnscaled['yGyrX'] - aPredUnscaled['yGyrX'].mean(), label='Roll Rate')
ax.scatter(aUnscaled[controlInput],aPredUnscaled['yGyrZ'] - aPredUnscaled['yGyrZ'].mean(), label='Yaw Rate')
ax.scatter(aUnscaled[controlInput],aPredUnscaled['yGyrY'] - aPredUnscaled['yGyrY'].mean(), label='Pitch Rate')
ax.set_ylabel('Rotation Rate (Degrees per Second)')
ax.set_xlabel('Elevator Deflection')
ax.set_title('ICE Plot - Rotation Rates over Elevator Deflection')
ax.legend()
ax.grid(alpha=.15)
plt.show()

# %%
# airspeed effect on control rate
# slow
controlInput = 'C1RCOU'
b = xTest.sample(1, random_state=3)
interestCols = ['AOA', 'SSA', 'Spd', 'Roll', 'Pitch', 'GyrX', 'C1RCOU']
b = pd.DataFrame(b.values.repeat(10, axis=0), columns=xCols)
b[controlInput] = np.linspace(xTest[controlInput].min(),xTest[controlInput].max(),10)
b['AspdE'] = 6
bUnscaled = pd.DataFrame(xScaler.inverse_transform(b), columns=xCols)

bPred = pd.DataFrame(model.predict(b), columns=y)
bPredUnscaled = pd.DataFrame(yScaler.inverse_transform(bPred), columns=y)
bPredUnscaled[['yGyrX','yGyrY','yGyrZ']] = bPredUnscaled[['yGyrX','yGyrY','yGyrZ']] * 180/np.pi 

# fast
c = xTest.sample(1, random_state=3)
interestCols = ['AOA', 'SSA', 'Spd', 'Roll', 'Pitch', 'GyrX', 'C1RCOU']
c = pd.DataFrame(c.values.repeat(10, axis=0), columns=xCols)
c[controlInput] = np.linspace(xTest[controlInput].min(),xTest[controlInput].max(),10)
c['AspdE'] = 11
cUnscaled = pd.DataFrame(xScaler.inverse_transform(c), columns=xCols)

cPred = pd.DataFrame(model.predict(c), columns=y)
cPredUnscaled = pd.DataFrame(yScaler.inverse_transform(cPred), columns=y)
cPredUnscaled[['yGyrX','yGyrY','yGyrZ']] = cPredUnscaled[['yGyrX','yGyrY','yGyrZ']] * 180/np.pi 

# plots
fig,ax = plt.subplots(dpi=100)
ax.scatter(bUnscaled[controlInput],bPredUnscaled['yGyrX'] - bPredUnscaled['yGyrX'].mean(), label='Roll Rate')
ax.scatter(bUnscaled[controlInput],bPredUnscaled['yGyrZ'] - bPredUnscaled['yGyrZ'].mean(), label='Yaw Rate')
ax.scatter(bUnscaled[controlInput],bPredUnscaled['yGyrY'] - bPredUnscaled['yGyrY'].mean(), label='Pitch Rate')
ax.scatter(cUnscaled[controlInput],cPredUnscaled['yGyrX'] - cPredUnscaled['yGyrX'].mean(), c='C0', marker='x')
ax.scatter(cUnscaled[controlInput],cPredUnscaled['yGyrZ'] - cPredUnscaled['yGyrZ'].mean(), c='C1', marker='x')
ax.scatter(cUnscaled[controlInput],cPredUnscaled['yGyrY'] - cPredUnscaled['yGyrY'].mean(), c='C2', marker='x')
ax.set_ylabel('Rotation Rate (Degrees per Second)')
ax.set_xlabel('Aileron Deflection')
ax.set_title('ICE Plot - Rotation Rates over Aileron Deflection')
ax.legend()
ax.grid(alpha=.15)
plt.show()
# %%
