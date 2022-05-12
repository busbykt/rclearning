'''
It may be useful to discretize the continuous state/action spaces
for a RL algorithm like Q-Learning.
'''

# %%
# dependencies
import os
import pandas as pd
import numpy as np

# %%
# ensure the working directory is rclearning
if os.path.basename(os.getcwd()) == 'rclearning':
    os.chdir('..')
assert os.path.basename(os.getcwd()) == 'rclearning'

# %%
# list features that may be worth discretizing
featuresToDiscretize = [
    'Roll', 
    'Pitch', 
    'Yaw', 
    'GyrX', 
    'GyrY', 
    'GyrZ', 
    'AccX',
    'AccY',
    'AccZ',
    'C1', # ail
    'C2', # elev
    'C3', # thr
    'C4', # rud
    'C1RCOU',
    'C2RCOU',
    'C3RCOU',
    'C4RCOU',   
    ]