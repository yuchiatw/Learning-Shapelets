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

def mul_to_binary(script, label):
    X = []
    y = []
    for i in range(len(label)):
            if label[i] == np.str_('1'):
                y.append(int(0))
                X.append(script[i])
            elif label[i] != np.str_('5'):
                y.append(int(1))
                X.append(script[i])
    return np.array(X), np.array(y)

def public_pipeline(
    dataset = 'ECG200',
    root = './data',
    config = {}, 
    output = False
):
    if not config:
        config = {
            'test_ratio': 0.2,
            'val_ratio': 0.2,
            'norm_std': 'standard',
        }
    
    script_all, label_all = load_classification(dataset)
    script_train, label_train = load_classification(dataset, split='train')
    script_test, label_test= load_classification(dataset, split='test')
    
    if dataset == 'ECG5000':
        X, y = mul_to_binary(script_all, label_all)
        X_train, y_train = mul_to_binary(script_train, label_train)
        X_test, y_test = mul_to_binary(script_test, label_test)
    else:
        X = script_all
        y = label_all
        X_train = script_train
        X_test = script_test
        y_train = label_train
        y_test = label_test
    
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=config['val_ratio'])
    X_train, scaler = normalize_data(X_train, mode=config['norm_std'])
    X_val, scaler = normalize_data(X_val, mode=config['norm_std'])
    X_test, scaler = normalize_data(X_test, mode=config['norm_std'])
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y = label_encoder.transform(y)
    y_train = label_encoder.transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)         
    if output:
        folder_name = os.path.join(root, dataset)
        print(folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        script_all_reshaped = np.reshape(X, (X.shape[0], -1))
        script_all_df = pd.DataFrame(script_all_reshaped)
        script_all_df.to_csv(os.path.join(folder_name, "script_all.csv"), index=False)
        # np.savetxt(os.path.join(folder_name, "script_all.csv"), script_all_reshaped, delimiter=",")
        np.savetxt(os.path.join(folder_name, "label_all.csv"), y, delimiter=",")
    data = {}
    data['X_train'] = X_train
    data['X_val'] = X_val
    data['X_test'] = X_test
    data['y_train'] = y_train
    data['y_val'] = y_val
    data['y_test'] = y_test
    
    # Save the dataset to disk
    data_path = os.path.join(root, f'{dataset}.npz')
    np.savez(data_path, **data)
    return data
if __name__ == '__main__':
    public_pipeline(dataset='ECG5000', output=True)
    a = pd.read_csv('data/ECG5000/script_all.csv')
    print(a.shape)
    # for rowNum, row in enumerate(a.values):
    #     if rowNum < 5:
    #         print(rowNum, row)