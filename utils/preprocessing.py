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

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, \
    roc_auc_score, precision_score, recall_score
from tslearn.clustering import TimeSeriesKMeans


from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt
from matplotlib import cm

import torch


def normalize_standard(X, mode='minmax', scaler=None):
    shape = X.shape
    data_flat = X.flatten()
    if scaler is None:
        scaler = MinMaxScaler()
        if mode == 'standard':
            scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data_flat.reshape(np.prod(shape), 1)).reshape(shape)
    else:
        data_transformed = scaler.transform(data_flat.reshape(np.prod(shape), 1)).reshape(shape)
    return data_transformed, scaler

def normalize_data(X, mode='minmax', glob_norm=False, scaler=None):
    '''
    Input:
    - X: time series data
    - mode: 'minmax' or 'standard'  
    - Scaler: if None, fit the scaler to the data
    '''
    if scaler is None:
        for i in range(X.shape[0]):
            X[i], scaler = normalize_standard(X[i], mode=mode)
    else:
        X, scaler = normalize_standard(X, scaler)
    
    return X, scaler

def segment_time_series(time_series, label, ID, segment_length = 200):
    n = len(time_series)
    # Number of complete segments
    num_segments = n // segment_length
    # Segment the series and assign the label to each segment
    segments = [
        time_series[i * segment_length:(i + 1) * segment_length]
        for i in range(num_segments)
    ]
    segment_labels = [label] * num_segments
    sgemet_ids = [ID] * num_segments
    return segments, segment_labels, sgemet_ids
def split_by_padding(time_series, padding_threshold):
    # Identify start and end indices of non-padding regions
    non_padding = np.where(time_series != 0, 1, 0)
    changes = np.diff(non_padding, prepend=0, append=0)
    start_indices = np.where(changes == 1)[0]
    end_indices = np.where(changes == -1)[0]
    
    # Filter out regions smaller than the padding threshold
    regions = [
        time_series[start:end]
        for start, end in zip(start_indices, end_indices)
        if end - start > padding_threshold
    ]
    return regions

# Function to segment a single time series
def segment_time_series_excluding_padding(time_series, label, ID, 
                                          segment_length, step = 100, 
                                          padding_threshold = 10):
    # Split into non-padding regions
    non_padding_regions = split_by_padding(time_series, padding_threshold)
    
    segments = []
    segment_labels = []
    segment_ids = []
    
    for region in non_padding_regions:
        # Segment each region
        region_segments = [
            region[i:i + segment_length]
            for i in range(0, len(region), step) if i + segment_length <= len(region)
        ]
        segments.extend(region_segments)
        segment_labels.extend([label] * len(region_segments))
        segment_ids.extend([ID] * len(region_segments))
    
    return segments, segment_labels, segment_ids
def segmentation(tocometer, labels, IDs, normal_first = True, seq_length = 100):
    all_segments = []
    all_labels = []
    all_ids = []
    for ts, label, ID in zip(tocometer, labels, IDs):
        if normal_first:
            ts, scaler = normalize_data(np.array(ts).reshape(1, -1))
            ts = ts.reshape(-1)
        segments, segment_labels, segments_ids = \
            segment_time_series_excluding_padding(ts, label, ID, segment_length=seq_length)
        
        if normal_first:
            all_segments.extend(segments)
        else:
            temp = [normalize_data(np.array(segments[i]).reshape(1, -1)) for i in range(len(segments))]
            print(temp[0].shape)
            all_segments.extend(temp)
        all_labels.extend(segment_labels)
        all_ids.extend(segments_ids)
    all_segments = np.array(all_segments).reshape(len(all_segments), 1, seq_length)
    return all_segments, all_labels, all_ids
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
def eval_accuracy(model, X, Y):
    predictions = model.predict(X)
    if len(predictions.shape) == 2:
        predictions = predictions.argmax(axis=1)
    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")
    return (predictions == Y).sum() / Y.size
    
def torch_dist_ts_shapelet(ts, shapelet, cuda=True):
    """
    Calculate euclidean distance of shapelet to a time series via \
        PyTorch and returns the distance along with the position in the time series.
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

def eval_results(y, yhat):
    if len(yhat.shape) == 2:
        yhat = yhat.argmax(axis=1)
        
    accuracy = accuracy_score(y, yhat)
    precision = precision_score(y, yhat)
    f1 = f1_score(y, yhat)
    recall = recall_score(y, yhat)
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'f1_score':f1, 
        'recall': recall,
    }
    