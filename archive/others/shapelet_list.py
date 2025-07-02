import os
import numpy as np
import argparse

from exp_pipeline import shapelet_initialization
from preterm_preprocessing.preterm_preprocessing import preterm_pipeline
from public_preprocessing.public_preprocessing import public_pipeline
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
# -----------------------------------
parser.add_argument('--datatype', type=str, default='private', choices={'public', 'private'})
parser.add_argument('--dataset', type=str, default='preterm')
parser.add_argument('--version', type=str, default='3')

# -----------------------------------


def shapelet_list_generator(config, datatype, dataset, version):

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
    else:
        data = public_pipeline(
            dataset=dataset, 
            output=False, 
            root='./data',
            config=config['data_loading'], 
        )
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    if config['init_mode'] == 'pips':
        print(config)
        shapelets_size_and_len, list_shapelets_meta, list_shapelets =\
            shapelet_initialization(X_train, y_train, 
                                    config=config['init_config'], 
                                    dataset=dataset, 
                                    mode=config['init_mode'], 
                                    version=version)
        
    else:
        shapelets_size_and_len = shapelet_initialization(X_train, y_train, config['init_config'], config['init_mode'] )
        
if __name__ == '__main__':
    args = parser.parse_args()
    config_command = args.__dict__
    config = { # default
        'init_mode': 'pips',
        'init_config': {
            'ws_rate': 0.05,
            'num_pip': 0.2, 
            'num_shapelets_make': 100, 
            'num_shapelets': 10,
        },
       
    }
    shapelet_list_generator(
        config, config_command['datatype'], 
        config_command['dataset'], config_command['version']
    )