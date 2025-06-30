import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())

from utils.preprocessing import normalize_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from aeon.datasets import load_classification
import pandas as pd
from synthetic.sample_generation import sythesize_data


def synthetic_pipeline(
    time_step = 10, 
    num_samples=1000,
    config = {}, 
    root = './data'
):
    if not config:
        config = {
            'test_ratio': 0.2,
            'val_ratio': 0.2,
            'norm_std': 'minmax',
        }
    
    X_train, y_train = sythesize_data(num_samples=num_samples, time_steps=time_step)
    X_val, y_val = sythesize_data(num_samples=int(config['val_ratio'] * num_samples), time_steps=time_step)
    X_test, y_test = sythesize_data(num_samples=int(config['test_ratio'] * num_samples), time_steps=time_step)
    X_train, scaler = normalize_data(X_train, mode=config['norm_std'])
    X_val, scaler = normalize_data(X_val, mode=config['norm_std'])
    X_test, scaler = normalize_data(X_test, mode=config['norm_std'])
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)
    
    data = {}
    data['X_train'] = X_train
    data['X_val'] = X_val
    data['X_test'] = y_test
    data['y_train'] = y_train
    data['y_val'] = y_val
    data['y_test'] = y_test
    
    # Save the dataset to disk
    data_path = os.path.join(root, f'synthetic_{num_samples}_{time_step}.npz')
    np.savez(data_path, **data)
    return data   