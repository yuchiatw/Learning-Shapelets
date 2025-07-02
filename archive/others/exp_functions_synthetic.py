import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

# import sys
# sys.path.insert(0, os.path.abspath('../'))
# sys.path.insert(0, os.getcwd())
print(f"Executing script at: {os.getcwd()}")
import numpy as np
import pandas as pd
import time
import random
import yaml
import glob

from torch import nn, optim
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from preterm_preprocessing.preterm_preprocessing import preterm_pipeline
from public_preprocessing.public_preprocessing import public_pipeline
from synthetic.synthetic_preprocessing import synthetic_pipeline
from Shapelet.mul_shapelet_discovery import ShapeletDiscover
from src.learning_shapelets import LearningShapelets as LearningShapeletsFCN
from src.learning_shapelets_sliding_window import LearningShapelets as LearningShapeletsTranformer
from src.vanilla_transformer import Vanilla
from src.fe_shape_joint import JointTraining
from pyts.classification import BOSSVS # only univariate format
from src.fe_shape_joint import feature_extraction_selection, extraction_pipeline
from numpy.lib.stride_tricks import sliding_window_view
import tsfel
from utils.evaluation_and_save import eval_results, save_results_to_csv
import torch
import json
import argparse

torch.cuda.set_device(1)
def torch_dist_ts_shapelet(ts, shapelet, cuda=True):
    """
    Calculate euclidean distance of shapelet to a time series via PyTorch and returns the distance along with the position in the time series.
    """
    if not isinstance(ts, torch.Tensor):
        ts = torch.tensor(ts, dtype=torch.float)
    if not isinstance(shapelet, torch.Tensor):
        shapelet = torch.tensor(shapelet, dtype=torch.float)
    if cuda:
        ts = ts.cuda()
        shapelet = shapelet.cuda()
    shapelet = torch.unsqueeze(shapelet, 0)
    # unfold time series to emulate sliding window
    ts = ts.unfold(2, shapelet.shape[2], 1).contiguous()
    # calculate euclidean distance
    dists = torch.cdist(ts, shapelet, p=2)/shapelet.shape[2]
    dists = torch.sum(dists, dim=1)
    # otherwise gradient will be None
    # hard min compared to soft-min from the paper
    d_min, d_argmin = torch.min(dists, 1)
    return (d_min.cpu().detach().numpy(), d_argmin.cpu().detach().numpy())

def shapelet_initialization(
    X_train, y_train, config, num_classes=2, dataset='preterm', mode='pips', version='', regenerate=False):
    _, n_channels, len_ts = X_train.shape
        
    ws_rate = config['ws_rate']
    num_pip = config['num_pip']
    num_shapelets_make = config['num_shapelets_make']
    num_shapelets=config['num_shapelets']
    
    elapsed = 0
    csv_path = f'./data/list_shapelets_meta_{dataset}_{ws_rate}_{num_pip}_{num_shapelets_make}.csv'
    if (len(version) > 0):
        csv_path = f'./data/list_shapelets_meta_{dataset}_{ws_rate}_{num_pip}_{num_shapelets_make}_v{version}.csv'
    print(csv_path)
    if os.path.exists(csv_path):
        df_shapelets_meta = pd.read_csv(csv_path)
        list_shapelets_meta = df_shapelets_meta.values
        
    else:
        t1 = time.time()
        shape = ShapeletDiscover(window_size=int(len_ts*ws_rate),num_pip=num_pip)
        shape.extract_candidate(X_train)
        shape.discovery(X_train, y_train)
        list_shapelets_meta = shape.get_shapelet_info(number_of_shapelet=num_shapelets_make)
        list_shapelets_meta = list_shapelets_meta[list_shapelets_meta[:, 3].argsort()[::-1]]
        elapsed = time.time() - t1

    df_shapelets_meta = pd.DataFrame(list_shapelets_meta, columns=['series_position', 'start_pos', 'end_pos', 'inforgain', 'label', 'dim'])
    df_shapelets_meta.to_csv(csv_path, index=False)
    print(len(df_shapelets_meta))
    # Filter out long shapelets
    max_shapelet_length = int(len_ts * 0.5)  # Example: filter out shapelets longer than 50% of the time series length
    # df_shapelets_meta = df_shapelets_meta[df_shapelets_meta['end_pos'] - df_shapelets_meta['start_pos'] <= max_shapelet_length]
    print(len(df_shapelets_meta))
    if num_shapelets > len(df_shapelets_meta):
        list_shapelets_meta = df_shapelets_meta.values
    else:
        every = int(num_shapelets/num_classes)
        list_shapelets_meta = df_shapelets_meta.values[:num_shapelets]
    list_shapelets = {}
    for i in range(list_shapelets_meta.shape[0] if list_shapelets_meta is not None else 0):
        shape_size = int(list_shapelets_meta[i, 2] - int(list_shapelets_meta[i, 1]))
        if shape_size not in list_shapelets:
            list_shapelets[shape_size] = [i]
        else:
            list_shapelets[shape_size].append(i)

    list_shapelets = {key: list_shapelets[key] for key in sorted(list_shapelets)} 
    shapelets_size_and_len = dict()
    for i in list_shapelets.keys():
        shapelets_size_and_len[i] = len(list_shapelets[i])
    return shapelets_size_and_len, list_shapelets_meta, list_shapelets, elapsed
def sample_ts_segments(X, shapelets_size, n_segments=1000):
    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    segments = np.empty((n_segments, n_channels, shapelets_size))
    for i, k in enumerate(samples_i):
        s = random.randint(0, len_ts - shapelets_size)
        segments[i] = X[k, :, s:s+shapelets_size]
    return segments
def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=1000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters
    
    
def train_ls(
    data: dict, 
    shapelets_size_and_len: dict, 
    init_mode: str, 
    model_mode: str, 
    list_shapelets: dict = {}, 
    list_shapelets_meta: np.ndarray = np.zeros(10), 
    store_results = False, 
    dataset = 'ECG200', 
    datatype = 'public',
    config={}, 
    version: str = ""
):
    print(config)
    X_train = data['X_train']
    y_train = data['y_train']
    _, n_channels, len_ts = X_train.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_train))
    window_size = max(shapelets_size_and_len.keys())
    if dataset == 'preterm': 
        window_size = window_size
    window_step = config['step']
    print(window_step)
    if os.path.exists(f'./data/{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy'):
        X_train_split_filtered = np.load(f'./data/{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy')
    else:
        x_train = X_train.transpose(0, 2, 1)
        num_train, len_ts, in_channels = x_train.shape
        x_train_split = sliding_window_view(x_train, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
        num_windows = x_train_split.shape[1]
        x_train_split = x_train_split.reshape(num_train * num_windows, window_size, in_channels)
        cfg_file = tsfel.get_features_by_domain()
            
        clean_selected_features = ['Area under the curve', 'Average power', 
                                    'ECDF Percentile', 'ECDF Percentile', 
                                    'LPCC', 'LPCC', 'LPCC', 'LPCC', 'LPCC', 'Max', 
                                    'Mean', 'Median', 'Median absolute deviation', 
                                    'Median absolute diff', 'Positive turning points', 
                                    'Root mean square', 'Slope', 'Spectral decrease', 
                                    'Spectral distance', 'Spectral roll-off', 'Spectral skewness', 
                                    'Spectral slope', 'Spectral spread', 'Standard deviation', 
                                    'Sum absolute diff', 'Variance', 'Wavelet absolute mean_12.5Hz', 
                                    'Wavelet absolute mean_2.78Hz', 'Wavelet absolute mean_25.0Hz', 
                                    'Wavelet absolute mean_3.12Hz', 'Wavelet absolute mean_3.57Hz', 
                                    'Wavelet absolute mean_4.17Hz', 'Wavelet absolute mean_5.0Hz', 
                                    'Wavelet absolute mean_6.25Hz', 'Wavelet absolute mean_8.33Hz', 
                                    'Wavelet energy_2.78Hz', 'Wavelet energy_3.12Hz', 'Wavelet energy_3.57Hz',
                                    'Wavelet energy_4.17Hz', 'Wavelet energy_5.0Hz', 'Wavelet energy_6.25Hz', 
                                    'Wavelet energy_8.33Hz', 'Wavelet standard deviation_12.5Hz', 
                                    'Wavelet standard deviation_25.0Hz', 'Wavelet standard deviation_3.12Hz', 
                                    'Wavelet standard deviation_3.57Hz', 'Wavelet standard deviation_4.17Hz', 
                                    'Wavelet standard deviation_5.0Hz', 'Wavelet standard deviation_6.25Hz', 
                                    'Wavelet standard deviation_8.33Hz', 'Wavelet variance_12.5Hz', 'Wavelet variance_2.78Hz', 
                                    'Wavelet variance_25.0Hz', 'Wavelet variance_3.12Hz', 'Wavelet variance_3.57Hz', 
                                    'Wavelet variance_4.17Hz', 'Wavelet variance_5.0Hz', 'Wavelet variance_6.25Hz', 
                                    'Wavelet variance_8.33Hz']
        # Disable all features in cfg_file first
        for domain in cfg_file.keys():
            for feature in cfg_file[domain]:
                cfg_file[domain][feature]["use"] = "no"  # Ensure correct format

        # Enable only the selected features
        for domain in cfg_file.keys():
            for feature in cfg_file[domain]:
                if feature in clean_selected_features:
                    cfg_file[domain][feature]["use"] =  "yes"
        X_train_split_filtered = tsfel.time_series_features_extractor(cfg_file, x_train_split)
        scaler = StandardScaler()
        X_train_split_filtered = scaler.fit_transform(X_train_split_filtered.values)
        X_train_split_filtered = X_train_split_filtered.reshape(num_train, num_windows, -1)
        np.save(f'./data/{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy', X_train_split_filtered)
        
    num_features = X_train_split_filtered.shape[-1]
    print(num_features)

    model = JointTraining(
        shapelets_size_and_len=shapelets_size_and_len,
        seq_len=len_ts, 
        in_channels=n_channels, 
        loss_func = loss_func, 
        mode = config['joint_mode'], 
        num_features=num_features, 
        window_size=window_size, 
        step=config['step'],
        nhead=config['nhead'], 
        num_layers=config['num_layers'],
        num_classes=num_classes, 
        to_cuda = True
    )
    if init_mode == 'pips':
        for i, key in enumerate(list_shapelets.keys() if list_shapelets is not None else [0, 0]):
            weights_block = []
            for j in list_shapelets[key] if list_shapelets is not None else [0]:
                weights_block.append(X_train[int(list_shapelets_meta[j, 0]), :, int(list_shapelets_meta[j, 1]):int(list_shapelets_meta[j, 2])])
            weights_block = np.array(weights_block)
            model.set_shapelet_weights_of_block(i, weights_block)
    t1 = time.time()
    model_path = config['model_path']
    optimizer = optim.Adam(model.model.parameters(), lr=config['lr'], weight_decay=config['wd'], eps=config['epsilon'])
    model.set_optimizer(optimizer)
   
    loss =  model.fit(
        X_train, X_train_split_filtered, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        shuffle=True, 
        model_path= model_path
    )
        
    elapsed = time.time() - t1
    return elapsed
def update_config(default_config, yaml_config):
    for key, value in yaml_config.items():
        if isinstance(value, dict) and key in default_config:
            update_config(default_config[key], value)  # Recursive update for nested dictionaries
        elif key in default_config:
            default_config[key] = value  # Update only existing keys

    
    
def exp(config, datatype = 'private', dataset='preterm', store_results = False, version: str = ""):
    '''
    Return: 
    - elapsed: time spent for training
    - results: experiment performance report
    - val_loss: validation loss
    '''
    # Data loading preprocessing pipeline
    
    if datatype == 'synthetic':
        num_samples = config['data_loading']['num_samples']
        time_step = config['data_loading']['time_step']
        data_path = os.path.join('./data',f'synthetic_{num_samples}_{time_step}.npz')
    data_path = data_path = os.path.join('./data', f'{dataset}.npz')
    meta_path='./data/filtered_clinical_data.csv'
    strip_path='./data/filtered_strips_data.json'
    if len(version) > 0 and datatype == 'private':
        data_path = os.path.join('./data', f'{dataset}_v{version}.npz')
        meta_path=f'./data/filtered_clinical_data_v{version}.csv'
        strip_path=f'./data/filtered_strips_data_v{version}.json' 
        

    if os.path.exists(data_path):
        data = np.load(data_path)
    elif datatype == 'private':
        data = preterm_pipeline(
            config=config['data_loading'], 
            meta_path=meta_path, 
            strip_path=strip_path,
            data_path=data_path
        )
    elif datatype == 'synthetic':
        data = synthetic_pipeline(
            time_step=config['data_loading']['time_step'],
            num_samples=config['data_loading']['num_samples'],
            config=config['data_loading'],
            root='./data'
        )
    else:
        data = public_pipeline(
            dataset=dataset, 
            output=store_results, 
            root='./data',
            config=config['data_loading'], 
        )

    
    print(data['X_train'].shape)
    print(data['X_val'].shape)    
    print(data['X_test'].shape)
    print(data['y_train'].shape)
    print(data['y_val'].shape)
    print(data['y_test'].shape)
    
    X_train = data['X_train']
    y_train = data['y_train']
    
    _, n_channels, len_ts = X_train.shape
    num_classes = len(set(y_train))
    print(config)
    shapelets_size_and_len, list_shapelets_meta, list_shapelets, shapetime =\
        shapelet_initialization(X_train, y_train, 
                                config=config['init_config'], 
                                dataset=dataset, 
                                num_classes=num_classes,
                                mode=config['init_mode'], 
                                version=version)
    elapsed = train_ls(
        data,
        shapelets_size_and_len=shapelets_size_and_len,
        list_shapelets=list_shapelets,
        list_shapelets_meta=list_shapelets_meta,
        init_mode='pips',
        model_mode=config['model_mode'],
        config=config['model_config'],
        store_results=store_results, 
        dataset=dataset, 
        datatype=datatype,
        version=version
    )
    
        
   
    
    return elapsed, shapetime
    
