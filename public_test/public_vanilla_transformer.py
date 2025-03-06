import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

import sys 
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
import glob
import yaml

import pickle
import json
import math
import random
import argparse
import time
import csv

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib import cm

import torch
from torch import tensor
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.vanilla_transformer import TimeSeriesTransformer
from utils.preprocessing import normalize_data
from utils.evaluation_and_save import eval_results, save_results_to_csv
from aeon.datasets import load_classification
import argparse

def load_yaml_config(filepath):
    if not filepath or not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return yaml.safe_load(f) or {}

def parse_args(config):
    parser = argparse.ArgumentParser(description='Experiment with vanilla transformer.')
    parser.add_argument("--dataset", type=str, default=config.get('dataset',"ECG200"))
    parser.add_argument('--num_epochs', type=int, default=config.get('num_epochs', 5000), help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 64), help='Batch size for training.')
    parser.add_argument('--nhead', type=int, default=config.get('nhead', 4), help='Number of heads for the transformer model.')
    parser.add_argument('--num_layers', type=int, default=config.get('num_layers', 2), help='Number of layers for the transformer model.')
    parser.add_argument('--lr', type=float, default=config.get('lr',1e-3), help='Learning rate for the optimizer.')
    parser.add_argument('--wd', type=float, default=config.get('wd', 1e-2), help='Weight decay for the optimizer.')
    parser.add_argument('--patience', type=int, default=config.get('patience', 1000), help='Patience for early stopping.')
    parser.add_argument('--min_delta', type=float, default=config.get('min_delta', 1e-6), help='Minimum change in validation loss to qualify as an improvement.')
    args = parser.parse_args()
    return args

def train(
    model,
    X, Y, 
    X_val = None,
    Y_val = None,
    loss_func = None, 
    optimizer = None, 
    num_epochs=1, 
    batch_size=256, 
    to_cuda = True, 
    shuffle=False, 
    drop_last=False, 
    patience=10, 
    min_delta=1e-6,
    model_path="./best_model.pth",
):
    
    if optimizer == None:
        raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")
    
    if not isinstance(X, torch.Tensor):
        X = tensor(X, dtype=torch.float32).contiguous()
    if not isinstance(Y, torch.Tensor):
        Y = tensor(Y, dtype=torch.long).contiguous()
    if to_cuda:
        X = X.cuda()
        Y = Y.cuda()
    if X_val is not None and Y_val is not None:
        if not isinstance(X_val, torch.Tensor):
            X_val = tensor(X_val, dtype=torch.float32).contiguous()
        if not isinstance(Y_val, torch.Tensor):
            Y_val = tensor(Y_val, dtype=torch.long).contiguous()
        if to_cuda:
            X_val = X_val.cuda()
            Y_val = Y_val.cuda()
    

    train_dataset = TensorDataset(X, Y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)    
    val_dataset = None
    val_loader = None
    if X_val is not None and Y_val is not None:
        val_dataset = TensorDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, )
    
    
    progress_bar = tqdm(range(num_epochs), disable=False)
    model.train()
    loss_list = []
    val_loss = []
    best_val_loss = float('inf')
    epochs_no_improve=0
    early_stop = False
    for epoch in progress_bar:
        batch_loss = []
        for j, (x, y) in enumerate(train_loader):
            yhat = model(x)
            loss = loss_func(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_loss.append(loss.item())
        loss_list.append(sum(batch_loss) / len(batch_loss))
        batch_loss = []
        if val_loader is not None:
            model.eval()
            batch_loss = []
            with torch.no_grad():
                for j, (x, y) in enumerate(val_loader):
                    yhat = model(x)
                    loss = loss_func(yhat, y)
                    batch_loss.append(loss.item())
            val_loss_epoch = sum(batch_loss) / len(batch_loss)
            val_loss.append(val_loss_epoch)
            
            # Early stopping
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_path)
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            model.train()
        
    return loss_list, val_loss   
            
def predict(
    model, 
    X,
    batch_size=256, 
    to_cuda=True
):
    if not isinstance(X, torch.Tensor):
        X = tensor(X, dtype=torch.float32).contiguous()
    if to_cuda:
        X = X.cuda()
    eval_dataset = TensorDataset(X)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    model.eval()
    result=None
    with torch.no_grad():
        for x in eval_loader:
            y_hat = model(x[0])
            y_hat = y_hat.cpu().detach()
            result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
    
    return result
def eval_accuracy(y, yhat):
    if len(yhat.shape) == 2:
        yhat = yhat.argmax(axis=1)
    return (y==yhat).sum() / y.size
        

def main(ID = 0, config = {}):
    
    args = parse_args(config)
    print(args)
    print(config)
    num_epochs = args.num_epochs
    batch_size=args.batch_size
    lr = args.lr
    wd=args.wd
    nhead = args.nhead
    num_layers = args.num_layers
    patience=args.patience
    min_delta=args.min_delta
    to_cuda=True
    model_path = config.get('model_path', './model/best_model.pth')
    dataset = config.get('dataset', 'ECG200')
    load_dataset = dataset
    if load_dataset == 'robot':
        load_dataset = "SonyAIBORobotSurface1"
    print(dataset)
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

    n_ts, n_channels, len_ts = x_train.shape
    
    
    num_classes = len(set(y_train))
    
    acc_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    train_loss_list = []
    val_loss_list = []
    elapsed_list = []
    for i in range(10):
        model = TimeSeriesTransformer(
            seq_len=len_ts,
            num_classes=num_classes,
            channels=n_channels,
            nhead=nhead,
            num_layers=num_layers,
            to_cuda=to_cuda
        )
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss_func = nn.CrossEntropyLoss()
        start = time.time()
        train_loss, val_loss = train(
            model, x_train, y_train, 
            X_val=x_val, Y_val=y_val,
            loss_func=loss_func, 
            optimizer=optimizer, 
            num_epochs=num_epochs, 
            batch_size=batch_size,
            to_cuda=to_cuda, 
            patience=patience,
            min_delta=min_delta,
            model_path=model_path,
        )
        elapsed = time.time() - start
        y_hat = None
        if os.path.exists(model_path):
            best_model = TimeSeriesTransformer(
                seq_len=len_ts,
                num_classes=num_classes,
                channels=n_channels,
                nhead=nhead,
                num_layers=num_layers,
                to_cuda=to_cuda
            )
            best_model.load_state_dict(torch.load(model_path, weights_only=True))
            y_hat = predict(best_model, x_test)
        else: 
            y_hat = predict(model, x_test)
        results = eval_results(y_test, y_hat)
        acc_list.append(results['accuracy'])
        precision_list.append(results['precision'])
        f1_list.append(results['f1_score'])
        recall_list.append(results['recall'])
        train_loss_list.append(train_loss[-1])
        val_loss_list.append(val_loss[-1])
        elapsed_list.append(elapsed)
        fig = plt.figure()
                
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
    
    avg_results = {
        'avg_accuracy': avg_acc,
        'avg_f1': avg_f1,
        'avg_recall': avg_recall,
        'avg_precision': avg_prec,
        'avg_val_loss': avg_loss,
        'elapsed_time': avg_elapsed
    }
    return args, avg_results
    
    
if __name__ == '__main__':
    results = []
    yaml_files = glob.glob('./yaml_configs_vanilla/*.yaml')
    for i, config_path in enumerate(yaml_files):
        config = load_yaml_config(config_path)
        args, avg_results = main(i, config=config)
        
        result = {}
        for key, value in vars(args).items():
            result[key] = value
        for key, value in avg_results.items():
            result[key] = value
        results.append(result)
    save_results_to_csv(results, filename='results_public_vanilla.csv')