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

from src.learning_shapelets_sliding_window import LearningShapelets
from aeon.datasets import load_classification
import argparse
import time
root = './'
# downsampling
def take_mean(series, window_size = 4):
    n = len(series)
    reshaped = np.array(series)[:n - n % window_size].reshape(-1, window_size)
    mean_value = reshaped.mean(axis=1)
    return mean_value    

def normalize_standard(X, scaler=None):
    shape = X.shape
    data_flat = X.flatten()
    if scaler is None:
        scaler = MinMaxScaler()
        data_transformed = scaler.fit_transform(data_flat.reshape(np.prod(shape), 1)).reshape(shape)
    else:
        data_transformed = scaler.transform(data_flat.reshape(np.prod(shape), 1)).reshape(shape)
    return data_transformed, scaler

def normalize_data(X, scaler=None):
    if scaler is None:
        for i in range(X.shape[0]):
            X[i], scaler = normalize_standard(X[i])
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
    parser.add_argument('--num_epochs', type=int, default=config.get('num_epochs', 100), help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 32), help='Batch size for training.')
    parser.add_argument('--nhead', type=int, default=config.get('nhead', 4), help='Number of heads for the transformer model.')
    parser.add_argument('--num_layers', type=int, default=config.get('num_layers', 2), help='Number of layers for the transformer model.')
    parser.add_argument('--lr', type=float, default=config.get('lr',1e-3), help='Learning rate for the optimizer.')
    parser.add_argument('--wd', type=float, default=config.get('wd', 1e-2), help='Weight decay for the optimizer.')
    parser.add_argument('--epsilon', type=float, default=config.get("epsilon",1e-7), help='Epsilon for the optimizer.')
    parser.add_argument('--dist_measure', type=str, default=config.get("epsilon", 'euclidean'), help='Distance measure for the shapelet model.')
    parser.add_argument('--num_shapelets_ratio', type=float, default=config.get("num_shapelets_ratio", 0.1), help='Number of shapelets as a ratio of the time series length.')
    parser.add_argument('--size_ratio', type=float, default=config.get("size_ratio", 0.15), help='Size of shapelets as a ratio of the time series length.')
    parser.add_argument('--step', type=int, default=config.get("step",10), help='Step size for sliding window.')
    parser.add_argument('--window_size', type=int, default=config.get("window_size", 30))
    args = parser.parse_args()
    
    return args
def train(index, configuration_path = "/ECG200/test1.yaml"):
    args = parse_args(configuration_path)
    print("transformer-base")
    print(args)
    dataset = args.dataset
    load_dataset = dataset
    if dataset == 'robot':
        load_dataset = "SonyAIBORobotSurface1"
    x, label = load_classification(load_dataset)
    x_train, x_test, label_train, label_test \
        = train_test_split(x, label, test_size = 0.3, shuffle=False, random_state=42)
    
    # print(x_train.shape, label_train.shape)
    # print(x_test.shape, label_test.shape)
    
    y_train = np.unique(label_train, return_inverse=True)[1]
    y_test = np.unique(label_test, return_inverse=True)[1]
    x_train, scaler = normalize_data(x_train)
    x_test, scaler = normalize_data(x_test)

    
    n_ts, n_channels, len_ts = x_train.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_train))
    
    size_ratio = args.size_ratio
    num_shapelets_ratio = args.num_shapelets_ratio
    dist_measure = args.dist_measure
    lr = args.lr
    wd = args.wd
    epsilon = args.epsilon
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    step = args.step
    nhead = args.nhead
    num_layers = args.num_layers
    num_shapelets = args.num_shapelets_ratio * len_ts
    if num_shapelets % args.nhead != 0:
        num_shapelets = int(num_shapelets // args.nhead) * args.nhead
    
    shapelets_size_and_len = {
        int(0.05 * len_ts): num_shapelets, 
        int(0.1 * len_ts): num_shapelets, 
    }
    
    print(len_ts)
    window_size = max(shapelets_size_and_len.keys())
    model = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len, 
                          seq_len= len_ts,
                          window_size = window_size,
                          in_channels = n_channels,
                          step=step,
                          num_classes = num_classes,
                          loss_func = loss_func,
                          nhead=nhead, 
                          num_layers=num_layers,
                          to_cuda = False,
                          verbose = 1,
                          dist_measure = dist_measure)
    
    for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len.items()):
        weights_block = get_weights_via_kmeans(x_train, shapelets_size, num_shapelets)
        model.set_shapelet_weights_of_block(i, weights_block)
    
    optimizer = optim.Adam(model.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
    model.set_optimizer(optimizer)
    output_dir = os.path.join(os.path.join(root, args.dataset), "results")
    os.makedirs(output_dir, exist_ok=True)
    loss = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, shuffle=False, drop_last=False)
    # fig = plt.figure()
    # plt.plot(loss)
    # plt.title('loss over training steps')
    # plt.show()
    # fig.savefig(os.path.join(output_dir, f'loss_config_{index}.png'))
    

    acc = eval_accuracy(model, x_test, y_test)
    shapelets = model.get_shapelets()

    shapelet_transform = model.transform(x_test)
    dist_s1 = shapelet_transform[:, 0]
    dist_s2 = shapelet_transform[:, 1]
    
    weights, biases = model.get_weights_linear_layer()
    
    
    # fig = plt.figure(facecolor='white')
    # fig.set_size_inches(20, 8)
    # gs = fig.add_gridspec(12, 8)
    # fig_ax1 = fig.add_subplot(gs[0:3, :4])
    # fig_ax1.set_title("First learned shapelet plotted (in red) on top of its 10 best matching time series.")
    # for i in np.argsort(dist_s1)[:10]:
    #     fig_ax1.plot(x_test[i, 0], color='black', alpha=0.5)
    #     _, pos = torch_dist_ts_shapelet(x_test[i], shapelets[0])
    #     fig_ax1.plot(lead_pad_shapelet(shapelets[0, 0], pos), color='#F03613', alpha=0.5)

    # fig_ax2 = fig.add_subplot(gs[0:3, 4:])
    # fig_ax2.set_title("Second learned shapelet plotted (in red) on top of its 10 best matching time series.")
    # for i in np.argsort(dist_s2)[:10]:
    #     fig_ax2.plot(x_test[i, 0], color='black', alpha=0.5)
    #     _, pos = torch_dist_ts_shapelet(x_test[i], shapelets[1])
    #     fig_ax2.plot(lead_pad_shapelet(shapelets[1, 0], pos), color='#F03613', alpha=0.5)


    # fig_ax3 = fig.add_subplot(gs[4:, :])
    # fig_ax3.set_title("The decision boundaries learned by the model to separate the four classes.")
    # color = {0: '#F03613', 1: '#7BD4CC', 2: '#00281F', 3: '#BEA42E'}
    # fig_ax3.scatter(dist_s1, dist_s2, color=[color[l] for l in y_test])

    # caption = """Shapelets learned for the FaceFour dataset of the UCR archive plotted on top of the best matching time series (top two pictures).
    #         And the corresponding learned decision boundaries of the linear classifier on top of the shapelet transformed test data (bottom picture)."""
    # plt.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=14)
    # plt.savefig(os.path.join(output_dir, f'learning_config_{index}.png'), facecolor=fig.get_facecolor(), bbox_inches="tight")


    # # Reshape x_test into 2D array
    # x_test_2d = x_test.reshape(x_test.shape[0], -1)

    # # Convert to DataFrame
    # df_x_test = pd.DataFrame(x_test_2d)

    # # Output as CSV file
    # df_x_test.to_csv(root+'x_test.csv', index=False)
    
    # # Convert test labels to DataFrame
    # df_y_test = pd.DataFrame(y_test, columns=['label'])

    # # Output as CSV file
    # df_y_test.to_csv(root+'y_test.csv', index=False)
    
    
    # # Reshape shapelets into 2D array
    # shapelets_2d = shapelets.reshape(shapelets.shape[0], -1)

    # # Convert to DataFrame
    # df_shapelets = pd.DataFrame(shapelets_2d)

    # # Output as CSV file
    # df_shapelets.to_csv(root+'shapelets.csv', index=False)
    
    # pos_record = []
    # for i in range(shapelet_transform.shape[1]):
    #     dist = shapelet_transform[:, i]
    #     pos_column = []
    #     for j in range(len(dist)):
    #         _, pos = torch_dist_ts_shapelet(x_test[j], shapelets[i])
    #         pos_column.append(pos)
        
    #     pos_record.append(pos_column)
        
    # # Convert pos_record to 2D array
    # pos_record_2d = np.array(pos_record).T

    # # Convert to DataFrame
    # df_pos_record = pd.DataFrame(pos_record_2d)

    # # Output as CSV file
    # df_pos_record.to_csv(root + 'pos_record.csv', index=False)
    
    # # Convert shapelet_transform to DataFrame
    # df_shapelet_transform = pd.DataFrame(shapelet_transform)

    # # Output as CSV file
    # df_shapelet_transform.to_csv(root + 'shapelet_transform.csv', index=False)
    
    return args, acc
def save_results_to_csv(results, filename="results.csv"):
        keys = results[0].keys()
        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
            
if __name__ == "__main__":
    results = []
    train(0)
    # yaml_files = glob.glob(os.path.join(root, "yaml_configs/*.yaml"))
    # for i, config_path in enumerate(yaml_files):
    #     start = time.time()
    #     args, acc = train(index = i, configuration_path=config_path)
    #     end = time.time()
    #     elapsed = end - start
    #     result = {     
    #         "index": i,
    #         "accuracy": acc,
    #         "elapsed_time": elapsed
    #     }
    #     for key, value in vars(args).items():
    #         result[key] = value
    #     results.append(result)

    #     print(f"Total elapsed time: {elapsed:.2f} seconds")
    #     print("-----------------")
        
    # save_results_to_csv(results, filename="results_public_trans.csv")
        