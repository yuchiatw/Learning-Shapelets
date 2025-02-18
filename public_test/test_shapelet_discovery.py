import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())

import numpy as np
from aeon.datasets import load_classification

from sklearn.model_selection import train_test_split

from Shapelet.shapelet_discovery import ShapeletDiscover
from utils.preprocessing import normalize_data

if __name__ == "__main__":
    
    load_dataset = 'ECG200'
    
    x, label = load_classification(load_dataset)
    x_train, x_test, label_train, label_test \
        = train_test_split(x, label, test_size=0.2, shuffle=False, random_state=42)
    x_train, x_val, label_train, label_val \
        = train_test_split(x_train, label_train, test_size=0.1, shuffle=False, random_state=42)
    
    y = np.unique(label, return_inverse=True)[1]
    y_train = np.unique(label_train, return_inverse=True)[1]
    y_val = np.unique(label_val, return_inverse=True)[1]
    y_test = np.unique(label_test, return_inverse=True)[1]
    x_train, scaler = normalize_data(x_train)
    x_val, scaler = normalize_data(x_val)
    x_test, scaler = normalize_data(x_test)
    
    shapelet_class = ShapeletDiscover()
    shapelet_class.extract_candidate(x_train)
    # print(shapelet_class.train_data_ci, shapelet_class.train_data_ci_piss)