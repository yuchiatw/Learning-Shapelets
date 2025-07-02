import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

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
from shapelet_candidate.mul_shapelet_discovery import ShapeletDiscover
from src.learning_shapelets import LearningShapelets as LearningShapeletsFCN
from src.learning_shapelets_sliding_window import LearningShapelets as LearningShapeletsTranformer
from src.vanilla_transformer import Vanilla
from src.fe_shape_joint import JointTraining
from pyts.classification import BOSSVS # only univariate format
from src.fe_shape_joint import feature_extraction_selection, extraction_pipeline
from numpy.lib.stride_tricks import sliding_window_view
import tsfel
from utils.evaluation_and_save import eval_results 
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
    X_train, y_train, config, num_classes=2, dataset='preterm', mode='pips', version=''
):
    _, n_channels, len_ts = X_train.shape
    if mode=='pips':
        
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
        max_shapelet_length = int(len_ts * 0.3)  # Example: filter out shapelets longer than 50% of the time series length
        df_shapelets_meta = df_shapelets_meta[df_shapelets_meta['end_pos'] - df_shapelets_meta['start_pos'] <= max_shapelet_length]
        print(len(df_shapelets_meta))
        if num_shapelets > len(df_shapelets_meta):
            list_shapelets_meta = df_shapelets_meta.values
        else:
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
    else:
        size_ratio = config['size_ratio']
        num_shapelets = config['num_shapelets']
        shapelets_size_and_len = dict()
        for i in size_ratio:
            size = int(len_ts * i)
            shapelets_size_and_len[size] = num_shapelets
        
        return shapelets_size_and_len
        
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
def store_data(data, dataset, model, list_shapelets_meta, list_shapelets):
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    X_all = np.concatenate((X_train, X_val, X_test), axis=0)
    y_all = np.concatenate((y_train, y_val, y_test), axis=0)
    shapelets = model.get_shapelets()
    i = 0
    output_shapelet = []
    for key in sorted(list_shapelets.keys()):
        for idx in list_shapelets[key]:
            shape_len = int(key)
            wave = shapelets[i, :, :shape_len]
            shape_info = {
                'len': shape_len,
                'gain': list_shapelets_meta[idx, 3],
                'wave': shapelets[i, :, :shape_len]
            }
            i += 1
            output_shapelet.append(shape_info)
    
    match_position = []
    min_distance = []
    for i in range(len(output_shapelet)):
        
        d_min, pos_start = torch_dist_ts_shapelet(X_all, output_shapelet[i]['wave'])
        pos_end = np.zeros(pos_start.shape)
        pos_end = pos_start + output_shapelet[i]['len']
        pos = np.concatenate((pos_start, pos_end), axis=1)
        pos = np.expand_dims(pos, 0)
        match_position.append(pos)
        min_distance.append(d_min)
    
    match_position = np.concatenate(match_position)
    min_distance = np.concatenate(min_distance, axis=1)
    print(min_distance.shape)
    match_position = np.transpose(match_position, (1, 0, 2))
    
    
    # Sort match_position based on output_shapelet['gain'] on axis 1
    gains = np.array([shapelet['gain'] for shapelet in output_shapelet])
    sorted_indices = np.argsort(gains)[::-1]
    match_position = match_position[:, sorted_indices, :]
    min_distance = min_distance[:, sorted_indices]
    
    match_position_start = match_position[:, :, 0].reshape(match_position.shape[0], match_position.shape[1])
    match_position_end = match_position[:, :, 1].reshape(match_position.shape[0], match_position.shape[1])
    # Sort output_shapelet based on 'gain'
    output_shapelet = sorted(output_shapelet, key=lambda x: x['gain'], reverse=True)
    output_shapelet = output_shapelet[:10]  # Keep only the top 10 shapelets based on gain
    output_dir = f"./data/{dataset}"
    
    os.makedirs(output_dir, exist_ok=True)
    min_distance_df = pd.DataFrame(min_distance[:, :10])
    min_distance_df.to_csv(os.path.join(output_dir, "shapelet_transform.csv"), index=False)
    X_all_df = pd.DataFrame(X_all.reshape(X_all.shape[0], -1))
    X_all_df.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    y_all_df = pd.DataFrame(y_all, columns=['label'])
    y_all_df.to_csv(os.path.join(output_dir, "label.csv"), index=False)
    match_start_df = pd.DataFrame(match_position_start[:, :10])
    match_start_df.to_csv(os.path.join(output_dir, "match_start.csv"), index=False)
    match_end_df = pd.DataFrame(match_position_end[:, :10])
    match_end_df.to_csv(os.path.join(output_dir, "match_end.csv"), index=False)
    output_shapelet_json = [
        {
            'len': shapelet['len'],
            'gain': shapelet['gain'],
            'wave': shapelet['wave'].tolist()
        }
        for shapelet in output_shapelet
    ]

    with open(os.path.join(output_dir, "output_shapelet.json"), 'w') as f:
        json.dump(output_shapelet_json, f)
    
    
def train_ls(
    data: dict, 
    shapelets_size_and_len: dict, 
    init_mode: str, 
    model_mode: str, 
    list_shapelets: dict = {}, 
    list_shapelets_meta: np.ndarray = np.zeros(10), 
    store_results = False, 
    dataset = 'ECG200', 
    config={}, 
    version: str = ""
):
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    _, n_channels, len_ts = X_train.shape
    loss_func = nn.CrossEntropyLoss()
    
    num_classes = len(set(y_train))
    if model_mode == 'LS_FCN':
        model = LearningShapeletsFCN(
            loss_func=loss_func,
            shapelets_size_and_len=shapelets_size_and_len, 
            in_channels=n_channels, 
            num_classes=num_classes,
            k=config['k'],
            l1=config['l1'],
            l2=config['l2'],
            to_cuda=True,
            verbose=1,
            dist_measure='euclidean',
        )
    elif model_mode == 'LS_Transformer':
        window_size = max(shapelets_size_and_len.keys())
        model = LearningShapeletsTranformer(
            shapelets_size_and_len=shapelets_size_and_len,
            loss_func=loss_func,
            seq_len=len_ts,
            in_channels=n_channels,
            num_classes=num_classes,
            k=config['k'],
            l1=config['l1'],
            l2=config['l2'],
            window_size=window_size,
            step=config['step'], 
            verbose=1,
            to_cuda=True
        )

    elif model_mode == 'JOINT':
        window_size = max(shapelets_size_and_len.keys())
        if dataset == 'preterm': 
            window_size = window_size
        window_step = config['step']
        if os.path.exists(f'./data/{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy'):
            X_train_split_filtered = np.load(f'./data/{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy')
            X_val_split_filtered = np.load(f'./data/{dataset}_X_val_split_filtered_{window_size}_{window_step}_{version}.npy')
            X_test_split_filtered = np.load(f'./data/{dataset}_X_test_split_filtered_{window_size}_{window_step}_{version}.npy')
        else:
            x_train = X_train.transpose(0, 2, 1)
            x_val = X_val.transpose(0, 2, 1)
            x_test = X_test.transpose(0, 2, 1)
            num_train, len_ts, in_channels = x_train.shape
            num_val = x_val.shape[0]
            num_test = x_test.shape[0]
            x_train_split = sliding_window_view(x_train, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
            x_val_split = sliding_window_view(x_val, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
            x_test_split = sliding_window_view(x_test, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
            num_windows = x_test_split.shape[1]
            x_train_split = x_train_split.reshape(num_train * num_windows, window_size, in_channels)
            x_val_split = x_val_split.reshape(num_val * num_windows, window_size, in_channels)
            x_test_split = x_test_split.reshape(num_test * num_windows, window_size, in_channels)
            cfg_file = tsfel.get_features_by_domain()
            if dataset == 'preterm':
                clean_selected_features = ['Average power', 'ECDF Percentile Count', 
                    'LPCC', 'LPCC', 'LPCC', 'LPCC', 'LPCC', 
                    'Median', 'Root mean square', 'Spectral distance', 
                    'Spectral roll-off', 'Spectral skewness', 'Spectral slope', \
                    'Spectral spread', 'Standard deviation', 'Sum absolute diff', 
                    'Wavelet absolute mean_25.0Hz', 'Wavelet absolute mean_3.12Hz', 
                    'Wavelet absolute mean_3.57Hz', 'Wavelet absolute mean_4.17Hz', 
                    'Wavelet absolute mean_5.0Hz', 'Wavelet absolute mean_6.25Hz', 
                    'Wavelet absolute mean_8.33Hz', 'Wavelet energy_25.0Hz', 'Wavelet energy_3.12Hz', 
                    'Wavelet energy_3.57Hz', 'Wavelet energy_4.17Hz', 'Wavelet energy_5.0Hz', 
                    'Wavelet energy_6.25Hz', 'Wavelet energy_8.33Hz', 'Wavelet standard deviation_12.5Hz', 
                    'Wavelet standard deviation_2.78Hz', 'Wavelet standard deviation_25.0Hz', 'Wavelet standard deviation_3.12Hz', 
                    'Wavelet standard deviation_3.57Hz', 'Wavelet standard deviation_4.17Hz', 'Wavelet standard deviation_5.0Hz', 
                    'Wavelet standard deviation_6.25Hz', 'Wavelet standard deviation_8.33Hz', 'Wavelet variance_2.78Hz', 
                    'Wavelet variance_3.12Hz', 'Wavelet variance_3.57Hz', 'Wavelet variance_4.17Hz', 'Wavelet variance_5.0Hz', 
                    'Wavelet variance_6.25Hz', 'Wavelet variance_8.33Hz'
                ]
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
                X_val_split_filtered = tsfel.time_series_features_extractor(cfg_file, x_val_split)
                X_test_split_filtered = tsfel.time_series_features_extractor(cfg_file, x_test_split)
                scaler = StandardScaler()
                X_train_split_filtered = scaler.fit_transform(X_train_split_filtered.values)
                X_val_split_filtered = scaler.transform(X_val_split_filtered.values)
                X_test_split_filtered = scaler.transform(X_test_split_filtered.values)
                X_train_split_filtered = X_train_split_filtered.reshape(num_train, num_windows, -1)
                X_val_split_filtered = X_val_split_filtered.reshape(num_val, num_windows, -1)
                X_test_split_filtered = X_test_split_filtered.reshape(num_test, num_windows, -1)
            else:
                
                X_train_split = tsfel.time_series_features_extractor(cfg_file, x_train_split)
                X_val_split = tsfel.time_series_features_extractor(cfg_file, x_val_split)
                X_test_split = tsfel.time_series_features_extractor(cfg_file, x_test_split)
                X_train_split_filtered, corr_features, selector, scaler = feature_extraction_selection(X_train_split)
                X_val_split_filtered = extraction_pipeline(X_val_split, corr_features, selector, scaler)
                X_test_split_filtered = extraction_pipeline(X_test_split, corr_features, selector, scaler)
                X_train_split_filtered = X_train_split_filtered.reshape(num_train, num_windows, -1)
                X_val_split_filtered = X_val_split_filtered.reshape(num_val, num_windows, -1)
                X_test_split_filtered = X_test_split_filtered.reshape(num_test, num_windows, -1)
            np.save(f'./data/{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy', X_train_split_filtered)
            np.save(f'./data/{dataset}_X_val_split_filtered_{window_size}_{window_step}_{version}.npy', X_val_split_filtered)
            np.save(f'./data/{dataset}_X_test_split_filtered_{window_size}_{window_step}_{version}.npy', X_test_split_filtered)
        
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
    else:
        for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len.items()):
            weights_block = get_weights_via_kmeans(X_train, shapelets_size, num_shapelets)
            model.set_shapelet_weights_of_block(i, weights_block)
    t1 = time.time()
    model_path = config['model_path']
    optimizer = optim.Adam(model.model.parameters(), lr=config['lr'], weight_decay=config['wd'], eps=config['epsilon'])
    model.set_optimizer(optimizer)
    if model_mode == 'LS_FCN' or model_mode == 'LS_Transformer':
        loss = model.fit(
            X_train, y_train, X_val=X_val, Y_val=y_val, shuffle=config['shuffle'],
            epochs=config['epochs'], batch_size=config['batch_size'],
            model_path=model_path
        )
    elif model_mode == 'JOINT':
        loss =  model.fit(
            X_train, X_train_split_filtered, y_train,
            X_val=data['X_val'], FE_val = X_val_split_filtered, Y_val=y_val,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            shuffle=True, 
            model_path= model_path
        )
        
    elapsed = time.time() - t1
    if os.path.exists(model_path):
        model.load_model(model_path)
    if model_mode == 'JOINT':
        y_hat = model.predict(X_test, FE = X_test_split_filtered)
    else:
        y_hat = model.predict(X_test)
    results = eval_results(y_test, y_hat)
    if store_results:
        store_data(data, dataset, model, list_shapelets_meta, list_shapelets)
    return elapsed, results, loss[-1]
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
    else:
        data_path = os.path.join('./data', f'{dataset}.npz')
    
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
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    _, n_channels, len_ts = X_train.shape
    num_classes = len(set(y_train))
    if config['model_mode'] == 'LS_FCN' \
        or config['model_mode'] == 'LS_Transformer' \
            or config['model_mode'] == 'JOINT':
        if config['init_mode'] == 'pips':
            print(config)
            shapelets_size_and_len, list_shapelets_meta, list_shapelets, shapetime =\
                shapelet_initialization(X_train, y_train, 
                                        config=config['init_config'], 
                                        dataset=dataset, 
                                        num_classes=num_classes,
                                        mode=config['init_mode'], 
                                        version=version)
            final_results = train_ls(
                data,
                shapelets_size_and_len=shapelets_size_and_len,
                list_shapelets=list_shapelets,
                list_shapelets_meta=list_shapelets_meta,
                init_mode='pips',
                model_mode=config['model_mode'],
                config=config['model_config'],
                store_results=store_results, 
                dataset=dataset, 
                version=version
            )
            elapsed = final_results[0]
            results = final_results[1]
            val_loss = final_results[2]
        else:
            shapelets_size_and_len = shapelet_initialization(X_train, y_train, config['init_config'], config['init_mode'] )
            elapsed, results, val_loss = train_ls(
                data, 
                shapelets_size_and_len=shapelets_size_and_len,
                init_mode='fixed_length',
                model_mode=config['model_mode'],
                config=config['model_config']
            )
    elif config['model_mode'] == 'BOSS':
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[2])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])
        window_size = config['model_config']['window_size']
        n_bins = config['model_config']['n_bins']
        window_step = config['model_config']['window_step']
        t1 = time.time()
        clf = BOSSVS(
            window_size=window_size, 
            window_step=window_step, 
            n_bins=n_bins,
        )
        clf.fit(X_train, y_train)
        elapsed = time.time() - t1
        y_val_pred = clf.predict(X_val)
        y_pred = clf.predict(X_test)
        val_loss = log_loss(y_val, y_val_pred)
        results = eval_results(y_test, y_pred)
        
    elif config['model_mode'] == 'vanilla':
        model_config = config['model_config']
        loss_func = nn.CrossEntropyLoss()
        model = Vanilla(
            loss_func=loss_func,
            seq_len=len_ts,
            num_classes=num_classes,
            channels=n_channels,
            num_layers=model_config['num_layers'],
            nhead=model_config['nhead'],
            d_model=model_config['d_model'],
            batch_first=True,
            to_cuda=True
        )
        optimizer = optim.Adam(
            model.model.parameters(), 
            lr=model_config['lr'], 
            weight_decay=model_config['wd'], 
            eps=model_config['epsilon']
        )
        model.set_optimizer(optimizer=optimizer)
        t1 = time.time()
        loss_list, val_loss = model.fit(
            X_train, y_train,
            X_val=X_val, Y_val=y_val,
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            shuffle=model_config['shuffle'],
            model_path=model_config['model_path']
        )
        elapsed = time.time() - t1
        model.load_model(model_config['model_path'])
        y_pred = model.predict(X_test)
        results = eval_results(y_test, y_pred)
        
   
    
    return elapsed, results, val_loss, shapetime
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with specified datatype and dataset.")
    parser.add_argument('--datatype', type=str, default='public', choices=['public', 'private', 'synthetic'], help='Type of data to use')
    parser.add_argument('--dataset', type=str, default='ECG5000', help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--model', type=str, default="JOINT", help='Batch size for training')
    parser.add_argument('--time_step', type=int, default=100, help='Time step for synthetic data (if applicable)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for synthetic data (if applicable)')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    print(args)
    datatype = args.datatype
    dataset = args.dataset
    if datatype == 'synthetic':
        dataset = f'synthetic_{args.num_samples}_{args.time_step}'
    batch_size= args.batch_size
    config = { # default
        'data_loading': {
            'time_step': args.time_step,  # For synthetic data only
            'num_samples': args.num_samples,  # For synthetic data only
            'test_ratio': 0.2,
            'val_ratio': 0.2,
            'norm_std': 'minmax',
            'norm_mode': 'local_before',
            'seq_min': 15,
            'pad_min': 3, 
            'step_min': 1,
        },
        'init_mode': 'pips',
        'model_mode': args.model,
        'init_config': {
            'ws_rate': 0.1,
            'num_pip': 0.2, 
            'num_shapelets_make': 500, 
            'num_shapelets': 10,
        },
        'model_config': {
            'window_size': 40,
            'window_step': 20,
            'n_bins': 4,
            'epochs': 1000, 
            'batch_size': batch_size, 
            'model_path': f'./model/best_model_{dataset}_new.pth',
            'joint_mode': 'concat', # concat / fusion
            'step': 10,
            'lr': 1e-3, 
            'wd': 1e-4, 
            'nhead': 2, 
            'd_model': 8,
            'num_layers': 2, 
            'epsilon': 1e-7,
            'shuffle': True,
            'k': 0,
            'l1': 0,
            'l2': 0
        },
    }

    
    report = []
    print(dataset)
    yaml_files = glob.glob("./preterm/yaml_JOINT/*.yaml")
    for config_path in yaml_files:
        try:
            with open(config_path, "r") as file:
                yaml_config = yaml.safe_load(file)  # Read YAML file
                # if yaml_config:  # Ensure it's not empty
                #     update_config(config, yaml_config)  # Update the config
        except FileNotFoundError:
            print(f"Warning: {config_path} not found. Using default config.")
    yaml_config = config
    elapsed, results, val_loss, shapetime = \
            exp(yaml_config, datatype=datatype, dataset=dataset, version='4', store_results=True)
    print(results)
    print(elapsed)
   