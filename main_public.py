import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import time
import random
import yaml
import glob

from aeon.datasets import load_classification

from torch import nn, optim
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import log_loss
from preterm_preprocessing.preterm_preprocessing import preterm_pipeline
from public_preprocessing.public_preprocessing import public_pipeline
from Shapelet.mul_shapelet_discovery import ShapeletDiscover
from src.learning_shapelets import LearningShapelets as LearningShapeletsFCN
from src.learning_shapelets_sliding_window import LearningShapelets as LearningShapeletsTranformer
from src.vanilla_transformer import TimeSeriesTransformer
from src.joint_training import MultiBranch
from pyts.classification import BOSSVS # only univariate format
import torch
torch.cuda.set_device(1)


# combine


from utils.evaluation_and_save import eval_results, save_results_to_csv

def shapelet_initialization(
    X_train, y_train, config, dataset='ECG200', mode='pips'
):
    _, n_channels, len_ts = X_train.shape
    if mode=='pips':
        
        ws_rate = config['ws_rate']
        num_pip = config['num_pip']
        num_shapelets_make = config['num_shapelets_make']
        num_shapelets=config['num_shapelets']
        
        if os.path.exists(f'./data/list_shapelets_meta_{dataset}_{ws_rate}_{num_pip}_{num_shapelets_make}.csv'):
            df_shapelets_meta = pd.read_csv(f'./data/list_shapelets_meta_{dataset}_{ws_rate}_{num_pip}_{num_shapelets_make}.csv')
            if num_shapelets > len(df_shapelets_meta):
                list_shapelets_meta = df_shapelets_meta.values
            else:
                list_shapelets_meta = df_shapelets_meta.values[:num_shapelets]
            
        else:
            t1 = time.time()
            shape = ShapeletDiscover(window_size=int(len_ts*ws_rate),num_pip=num_pip)
            shape.extract_candidate(X_train)
            shape.discovery(X_train, y_train)
            list_shapelets_meta = shape.get_shapelet_info(number_of_shapelet=num_shapelets_make)
            print(time.time() - t1)
        
            df_shapelets_meta = pd.DataFrame(list_shapelets_meta, columns=['series_position', 'start_pos', 'end_pos', 'inforgain', 'label', 'dim'])
            df_shapelets_meta.to_csv(f'./data/list_shapelets_meta_{dataset}_{ws_rate}_{num_pip}_{num_shapelets_make}.csv', index=False)
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
        return shapelets_size_and_len, list_shapelets_meta, list_shapelets
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

def train_ls(
    data: dict, 
    shapelets_size_and_len: dict, 
    init_mode: str, 
    model_mode: str, 
    list_shapelets: dict = {}, 
    list_shapelets_meta: np.ndarray = np.zeros(10), 
    config={}
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
    if model_mode == 'LT_FCN':
        model = LearningShapeletsFCN(
            loss_func=loss_func,
            shapelets_size_and_len=shapelets_size_and_len, 
            in_channels=n_channels, 
            num_classes=num_classes,
            to_cuda=True,
            verbose=1,
            dist_measure='euclidean'
        )
    elif model_mode == 'LT_Transformer':
        window_size = max(shapelets_size_and_len.keys())
        model = LearningShapeletsTranformer(
            shapelets_size_and_len=shapelets_size_and_len,
            loss_func=loss_func,
            seq_len=len_ts,
            in_channels=n_channels,
            num_classes=num_classes,
            window_size=window_size,
            step=config['step'], 
            verbose=1,
            to_cuda=True
        )
    elif model_mode == 'JOINT':
        window_size = max(shapelets_size_and_len.keys())
        model = MultiBranch(
            shapelets_size_and_len=shapelets_size_and_len,
            loss_func=loss_func,
            seq_len=len_ts,
            in_channels=n_channels,
            num_classes=num_classes,
            window_size=window_size,
            step=config['step'], 
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            verbose=1,
            to_cuda=True
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
            print(weights_block.shape)
            model.set_shapelet_weights_of_block(i, weights_block)
    t1 = time.time()
    model_path = config['model_path']
    optimizer = optim.Adam(model.model.parameters(), lr=config['lr'], weight_decay=config['wd'], eps=config['epsilon'])
    model.set_optimizer(optimizer)
    loss, val_loss = model.fit(
        X_train, y_train, X_val=X_val, Y_val=y_val, 
        epochs=config['epochs'], batch_size=config['batch_size'],
        model_path=model_path
    )
    elapsed = time.time() - t1
    if os.path.exists(model_path):
        if model_mode == 'LT_FCN':
            model = LearningShapeletsFCN(
                loss_func=loss_func,
                shapelets_size_and_len=shapelets_size_and_len, 
                in_channels=n_channels, 
                num_classes=num_classes,
                to_cuda=True,
                verbose=1,
                dist_measure='euclidean'
            )
        elif model_mode == 'LT_Transformer':
            model = LearningShapeletsTranformer(
                shapelets_size_and_len=shapelets_size_and_len,
                loss_func=loss_func,
                seq_len=len_ts,
                in_channels=n_channels,
                num_classes=num_classes,
                window_size=window_size,
                step=config['step'], 
                verbose=1,
                to_cuda=True,
            )
        elif model_mode == 'JOINT':
            model = MultiBranch(
            shapelets_size_and_len=shapelets_size_and_len,
            loss_func=loss_func,
            seq_len=len_ts,
            in_channels=n_channels,
            num_classes=num_classes,
            window_size=window_size,
            step=config['step'], 
            verbose=1,
            to_cuda=True
        )
        model.load_model(model_path)
    y_hat = model.predict(X_test)
    results = eval_results(y_test, y_hat)

    return elapsed, results, val_loss
def update_config(default_config, yaml_config):
    for key, value in yaml_config.items():
        if isinstance(value, dict) and key in default_config:
            update_config(default_config[key], value)  # Recursive update for nested dictionaries
        elif key in default_config:
            default_config[key] = value  # Update only existing keys

def exp(config, datatype = 'public', dataset='ECG5000', store_results = False):
    '''
    Return: 
    - elapsed: time spent for training
    - results: experiment performance report
    - val_loss: validation loss
    '''
    # Data loading preprocessing pipeline
    data_path = os.path.join('./data', f'{dataset}.npz')

    if os.path.exists(data_path):
        data = np.load(data_path)
    elif datatype == 'private':
        data = preterm_pipeline(
            config=config['data_loading'], 
            meta_path='./data/filtered_clinical_data_v2.csv', 
            strip_path='./data/filtered_strips_data_v2.json', 
            data_path=data_path
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
    if config['model_mode'] == 'LT_FCN' \
        or config['model_mode'] == 'LT_Transformer' \
            or config['model_mode'] == 'JOINT':
        if config['init_mode'] == 'pips':
            print(config)
            shapelets_size_and_len, list_shapelets_meta, list_shapelets =\
                shapelet_initialization(X_train, y_train, config['init_config'], config['init_mode'])
            elapsed, results, val_loss = train_ls(
                data,
                shapelets_size_and_len=shapelets_size_and_len,
                list_shapelets=list_shapelets,
                list_shapelets_meta=list_shapelets_meta,
                init_mode='pips',
                model_mode=config['model_mode'],
                config=config['model_config'],
            )
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
    return elapsed, results, val_loss
        
    # elif config['model_mode'] == 'vanilla':
    #     model_config = config['model_config']
    #     model = TimeSeriesTransformer(
    #         seq_len=len_ts, 
    #         num_classes=num_classes,
    #         channels=n_channels,
    #         d_model=model_config['d_model'], 
    #         nhead=model_config['nhead'],
    #         num_layers=model_config['num_layers'],
    #         to_cuda=True            
    #     )

if __name__=='__main__':
    # configuration loading
    config = { # default
        'data_loading': {
            'test_ratio': 0.2,
            'val_ratio': 0.2,
            'norm_std': 'standard',
            'norm_mode': 'local_before',
            'seq_min': 15,
            'pad_min': 3, 
            'step_min': 1,
        },
        'init_mode': 'pips',
        'model_mode': 'LT_Transformer', # 'JOINT' / 'LT_FCN' / 'LT_Transformer' / 'BOSS'
        'init_config': {
            'ws_rate': 0.1,
            'num_pip': 0.1, 
            'num_shapelets_make': 100, 
            'num_shapelets': 10,
        },
        # 'init_config': {
        #     'size_ratio': [0.1, 0.2], 
        #     'num_shapelets': 10
        # },
        'model_config': {
            'epochs': 300, 
            'batch_size': 256, 
            'model_path': './model/best_model',
            'step': 1,
            'lr': 1e-3, 
            'wd': 1e-4, 
            'epsilon': 1e-7
        },
    }
    
    dataset = 'ECG200'
    report = []
    
    yaml_files = glob.glob("./public_preprocessing/yaml_BOSS/*.yaml")
    print(yaml_files)
    yaml_config = config
    # for config_path in yaml_files:
        # try:
        #     with open(config_path, "r") as file:
        #         yaml_config = yaml.safe_load(file)  # Read YAML file
        #         # if yaml_config:  # Ensure it's not empty
        #         #     update_config(config, yaml_config)  # Update the config
        # except FileNotFoundError:
        #     print(f"Warning: {config_path} not found. Using default config.")
    
    for k in range(1):
        acc_list = []
        f1_list = []
        recall_list = []
        precision_list = []
        val_loss_list = []
        elapsed_list = []
        for j in range(10):
            elapsed, results, val_loss = \
                exp(yaml_config, datatype='public', dataset=dataset)
            acc_list.append(results['accuracy'])
            precision_list.append(results['precision'])
            f1_list.append(results['f1_score'])
            recall_list.append(results['recall'])
            val_loss_list.append(val_loss)
            elapsed_list.append(elapsed)
        
        avg_acc = sum(acc_list) / len(acc_list)
        avg_prec = sum(precision_list) / len(precision_list)
        avg_f1 = sum(f1_list) / len(f1_list)
        avg_recall = sum(recall_list) / len(recall_list)
        avg_loss = sum(val_loss_list) / len(val_loss_list)
        avg_elapsed = sum(elapsed_list) / len(elapsed_list)

        print(f"Average accuracy: {avg_acc}")
        print(f"Average precision: {avg_prec}")
        print(f"Average f1-score: {avg_f1}")
        print(f"Average recall score: {avg_recall}")
        print(f"Average validation loss: {avg_loss}")

        result = {
            'avg_accuracy': avg_acc,
            'avg_f1': avg_f1,
            'avg_recall': avg_recall,
            'avg_precision': avg_prec,
            'avg_val_loss': avg_loss,
            'elapsed_time': avg_elapsed
        }
        # result['init_mode'] = config['init_mode']
        result['model_mode'] = yaml_config['model_mode']
        for key, value in yaml_config['data_loading'].items():
            result[f'data_{key}'] = value
        for key, value in yaml_config['init_config'].items():
            result[f'init_{key}'] = value
        for key, value in yaml_config['model_config'].items():
            result[f'model_{key}'] = value
        report.append(result)
        print("-----------------")
    output_dir = f"./log/{dataset}/"
    os.makedirs(output_dir, exist_ok=True)
    save_results_to_csv(report, filename=os.path.join(output_dir, 'LS_Trans.csv'))