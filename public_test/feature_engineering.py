import os
import sys 
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())


import numpy as np
from pyts.datasets import load_gunpoint
from pyts.classification import TSBF, BOSSVS
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split
from utils.preprocessing import normalize_data
import time

# Toy dataset
X_train, y_train = load_classification("SonyAIBORobotSurface1")
X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
X_train, scaler = normalize_data(X_train, mode='standard')
X_test, scaler = normalize_data(X_test, mode='standard')
window_size = 24
n_bins = 8
window_step = 1

time_log = []
acc_log = []

clf = BOSSVS(
    window_size=window_size, 
    window_step=window_step, 
    n_bins=n_bins,
)
t1 = time.time()
clf.fit(X_train, y_train)
time_log.append(time.time() - t1)
acc_log.append(clf.score(X_test, y_test))

print(f'execution time is {sum(time_log)/len(time_log)}')
print(f'average acc is {sum(acc_log)/len(acc_log)}')