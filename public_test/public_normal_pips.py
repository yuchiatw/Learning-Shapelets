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

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tslearn.clustering import TimeSeriesKMeans

from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt
from matplotlib import cm

import torch
from torch import nn, optim

from src.learning_shapelets import LearningShapelets, LearningShapeletsModel
from Shapelet.mul_shapelet_discovery import ShapeletDiscover
from utils.preprocessing import normalize_data
from utils.evaluation_and_save import eval_results
from aeon.datasets import load_classification
import argparse
root = './'


def eval_accuracy(model, X, Y):
    predictions = model.predict(X)
    if len(predictions.shape) == 2:
        predictions = predictions.argmax(axis=1)
    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")
    return (predictions == Y).sum() / Y.size
    
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
    ts = ts.unfold(1, shapelet.shape[2], 1)
    # calculate euclidean distance
    dists = torch.cdist(ts, shapelet, p=2)
    dists = torch.sum(dists, dim=0)
    # otherwise gradient will be None
    # hard min compared to soft-min from the paper
    d_min, d_argmin = torch.min(dists, 0)
    return (d_min.item(), d_argmin.item())

def lead_pad_shapelet(shapelet, pos):
    """
    Adding leading NaN values to shapelet to plot it on a time series at the best matching position.
    """
    pad = np.empty(pos)
    pad[:] = np.nan
    padded_shapelet = np.concatenate([pad, shapelet])
    return padded_shapelet
def load_yaml_config(filepath):
    if not filepath or not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return yaml.safe_load(f) or {}
    
def parse_args(configuration_path = "test.yaml"):
    config = load_yaml_config(configuration_path)
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--dataset", type=str, default=config.get('dataset',"ECG200"))
    parser.add_argument('--num_epochs', type=int, default=config.get('num_epochs', 1000), help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 8), help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=config.get('lr',1e-3), help='Learning rate for the optimizer.')
    parser.add_argument('--wd', type=float, default=config.get('wd', 1e-3), help='Weight decay for the optimizer.')
    parser.add_argument('--epsilon', type=float, default=config.get("epsilon",1e-7), help='Epsilon for the optimizer.')
    parser.add_argument('--dist_measure', type=str, default=config.get("epsilon", 'euclidean'), help='Distance measure for the shapelet model.')
    parser.add_argument('--normal_mode', type=str, default=config.get("normal_mode", 'minmax'))
    parser.add_argument('--num_shapelet', type=int, default=config.get("num_shapelet", 10))
    parser.add_argument('--num_pip', type=float, default=config.get("num_pip", 0.1))
    parser.add_argument('--window_size_ratio', type=float, default=config.get("window_size_ratio", 0.1))
    args = parser.parse_args()
    
    return args
def train(index, configuration_path = "/ECG200/test.yaml"):
    args = parse_args(configuration_path)
    print("FCN-based")
    print(args)
    model_path = os.path.join('./model', args.dataset+"_"+str(index)+".pth")
    print(model_path)
    dataset = args.dataset
    load_dataset = dataset
    if dataset == 'robot':
        load_dataset = "SonyAIBORobotSurface1"
    x, label = load_classification(load_dataset)
    x_train, x_test, label_train, label_test \
        = train_test_split(x, label, test_size = 0.3, shuffle=False, random_state=42)
    x_train, x_val, label_train, label_val \
        = train_test_split(x_train, label_train, test_size=0.1, shuffle=False, random_state=42)
    
    y = np.unique(label, return_inverse=True)[1]
    y_train = np.unique(label_train, return_inverse=True)[1]
    y_val = np.unique(label_val, return_inverse=True)[1]
    y_test = np.unique(label_test, return_inverse=True)[1]
    x_train, scaler = normalize_data(x_train, mode=args.normal_mode)
    x_val, scaler = normalize_data(x_val, mode=args.normal_mode)
    x_test, scaler = normalize_data(x_test, mode=args.normal_mode)

    
    n_ts, n_channels, len_ts = x_train.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_train))
    start = time.time()
    shape = ShapeletDiscover(window_size=int(args.window_size_ratio * len_ts), num_pip=args.num_pip)
    shape.extract_candidate(x_train)
    shape.discovery(x_train, y_train)
    list_shapelets_meta = shape.get_shapelet_info(number_of_shapelet=args.num_shapelet)
    list_shapelets = {}
    for i in range(list_shapelets_meta.shape[0] if list_shapelets_meta is not None else 0):
        shape_size =  int(list_shapelets_meta[i, 2]) - int(list_shapelets_meta[i, 1])
        if shape_size not in list_shapelets:
            list_shapelets[shape_size] = [i]
        else:
            list_shapelets[shape_size].append(i)
    list_shapelets = {key: list_shapelets[key] for key in sorted(list_shapelets)}
    shapelets_size_and_len = dict()
    for i in list_shapelets.keys():
        shapelets_size_and_len[i] = len(list_shapelets[i])
    dist_measure = args.dist_measure
    model = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len, 
                          in_channels = n_channels,
                          num_classes = num_classes,
                          loss_func = loss_func,
                          to_cuda = True,
                          verbose = 1,
                          dist_measure = 'euclidean')
    for i, key in enumerate(list_shapelets.keys()):
        weights_block = []
        for j in list_shapelets[key]:
            weights_block.append(x_train[int(list_shapelets_meta[j, 0]), :, int(list_shapelets_meta[j, 1]):int(list_shapelets_meta[j, 2])])
        weights_block = np.array(weights_block)
        model.set_shapelet_weights_of_block(i, weights_block)

    lr = args.lr
    wd = args.wd
    epsilon = args.epsilon
    num_epochs = args.num_epochs
    batch_size = args.batch_size
        

    optimizer = optim.Adam(model.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
    model.set_optimizer(optimizer)
    loss, val_loss = model.fit(x_train, y_train, X_val=x_val, Y_val=y_val, 
                            epochs=num_epochs, batch_size=batch_size, shuffle=False, drop_last=False, 
                            model_path=model_path)
    elapsed = time.time() - start
    y_hat = None
    if os.path.exists(model_path):
        best_model = LearningShapelets(
            shapelets_size_and_len=shapelets_size_and_len, 
            in_channels = n_channels,
            num_classes = num_classes,
            loss_func = loss_func,
            to_cuda = True,
            verbose = 1,
            dist_measure = dist_measure
        )
        best_model.load_model(model_path)
        y_hat = best_model.predict(x_test)
    else: 
        y_hat = model.predict(x_test)
    results = eval_results(y_test, y_hat)

    return elapsed, args, results, val_loss

def save_results_to_csv(results, filename="results.csv"):
        keys = results[0].keys()
        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
            
if __name__ == "__main__":
    avg_results = []
    
    yaml_files = glob.glob(os.path.join(root, "yaml_configs_pips/*.yaml"))
    for i, config_path in enumerate(yaml_files):
        acc_list = []
        f1_list = []
        recall_list = []
        precision_list = []
        val_loss_list = []
        elapsed_list = []
        for j in range(10):
            elapsed, args, results, val_loss = train(index=i, configuration_path=config_path)
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
        for key, value in vars(args).items():
            result[key] = value
        avg_results.append(result)
        print("-----------------")
    save_results_to_csv(avg_results, filename="public_fcn_pips_init.csv")