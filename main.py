import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

import sys 
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
import glob
import yaml

import random
import argparse
import time
import csv


from utils.preprocessing import normalize_data
from utils.dataloader import load_data
import torch
def load_yaml_config(filepath):
    if not filepath or not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return yaml.safe_load(f) or {}
    
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument("--dataset", type=str, default="ECG200"))
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--wd', type=float, default=1e-3, help='Weight decay for the optimizer.')
    parser.add_argument('--epsilon', type=float, default=1e-7, help='Epsilon for the optimizer.')
    parser.add_argument('--dist_measure', type=str, default='euclidean', help='Distance measure for the shapelet model.')
    parser.add_argument('--num_shapelets_ratio', type=float, default=0.3, help='Number of shapelets as a ratio of the time series length.')
    parser.add_argument('--size_ratio', type=float, default= [0.125, 0.2], help='Size of shapelets as a ratio of the time series length.')
    parser.add_argument('--model_path', type=str, default='./model/model.pth', help='Folder to save the results.')
    args = parser.parse_args()
    
    if args.config:
        config = load_yaml_config(args.config)
        for key, value in config.items():
            parser.set_defaults(**{key: value})
    
    return args

if __name__=='__main__':
    config = parse_args()
    print(config)
    
