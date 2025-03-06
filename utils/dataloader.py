import os
import numpy as np
import logging
import random
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from aeon.datasets import load_classification

logger = logging.getLogger(__name__)

def load_data(config):
    Data = {}
    dataset_name = config['dataset']
    
    logger.info("Loading and preprocessing data ...")
    X, y = load_classification(dataset_name, split='train')
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=config['test_ratio'])

    y_train = LabelEncoder().fit_transform(y_train)
    y_test = LabelEncoder().fit_transform(y_test)

    max_seq_len = X_train.shape[2]

    if config['Norm']:
        mean, std = mean_std(X_train)
        mean = np.repeat(mean, max_seq_len).reshape(X_train.shape[1], max_seq_len)
        std = np.repeat(std, max_seq_len).reshape(X_train.shape[1], max_seq_len)
        X_train = mean_std_transform(X_train, mean, std)
        X_test = mean_std_transform(X_test, mean, std)
    
    if config['val_ratio'] > 0:
        train_data, train_label, val_data, val_label = split_dataset(X_train, y_train, config['val_ratio'])
    else:
        val_data, val_label = [None, None]
        
    Data['max_len'] = max_seq_len
    Data['All_data'] = X
    Data['All_label'] = y
    Data['train_data'] = train_data
    Data['train_label'] = train_label
    Data['val_data'] = val_data
    Data['val_label'] = val_label
    Data['test_data'] = X_test
    Data['test_label'] = y_test

    train_len = len(Data['train_label']) if isinstance(Data['train_label'], (list, np.ndarray)) else 0
    logger.info("{} samples will be used for training".format(train_len))
    val_len = len(Data['val_label']) if isinstance(Data['val_label'], (list, np.ndarray)) else 0
    logger.info("{} samples will be used for validation".format(val_len))
    test_len = len(Data['test_label']) if isinstance(Data['test_label'], (list, np.ndarray)) else 0
    logger.info("{} samples will be used for testing".format(test_len))

def split_dataset(data, label, validation_ratio):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]
    
    return train_data, train_label, val_data, val_label
    
def mean_std(train_data):
    m_len = np.mean(train_data, axis=2)
    mean = np.mean(m_len, axis=0)

    s_len = np.std(train_data, axis=2)
    std = np.max(s_len, axis=0)

    return mean, std


def mean_std_transform(train_data, mean, std):
    '''
    Normalizing based on global mean and std.
    '''
    return (train_data - mean) / std