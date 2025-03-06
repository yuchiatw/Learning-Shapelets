import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

import sys 
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np

import json
import random
import argparse
import yaml
import glob
import time

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
import csv


mean_window_size = 4
seq_min = 15
seq_length = int(60 * 4 / mean_window_size * seq_min)
padding_min = 3
padding_threshold = int(60 * 4 / mean_window_size * padding_min)
step_min = 1
step = int(60 * 4 / mean_window_size * step_min)

dist_measure = 'euclidean'
lr = 1e-3
wd = 1e-2
epsilon = 1e-7
num_epochs = 1000
num_layers = 2
nhead = 5

root = "./"
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
                                          segment_length, step = step, 
                                          padding_threshold = padding_threshold):
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
def segmentation(tocometer, labels, IDs, normal_first = True, seq_length = seq_length):
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
def sample_ts_segments(X, shapelets_size, n_segments=10000):
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
def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):
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
def parse_args(configuration_path="test.yaml"):
    config = load_yaml_config(configuration_path)

    parser = argparse.ArgumentParser(description='Experiment with preterm birth data.')
    parser.add_argument('--num_epochs', type=int, default=config.get('num_epochs', 1000), help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 32), help='Batch size for training.')
    parser.add_argument('--nhead', type=int, default=config.get('nhead', 4), help='Number of heads for the transformer model.')
    parser.add_argument('--num_layers', type=int, default=config.get('num_layers', 2), help='Number of layers for the transformer model.')
    parser.add_argument('--lr', type=float, default=config.get('lr',1e-3), help='Learning rate for the optimizer.')
    parser.add_argument('--wd', type=float, default=config.get('wd', 1e-2), help='Weight decay for the optimizer.')
    parser.add_argument('--epsilon', type=float, default=config.get("epsilon",1e-7), help='Epsilon for the optimizer.')
    parser.add_argument('--dist_measure', type=str, default=config.get("epsilon", 'euclidean'), help='Distance measure for the shapelet model.')
    parser.add_argument('--num_shapelets_ratio', type=float, default=config.get("num_shapelets_ratio", 0.1), help='Number of shapelets as a ratio of the time series length.')
    parser.add_argument('--size_ratio', type=float, default=config.get("size_ratio", 0.25), help='Size of shapelets as a ratio of the time series length.')
    parser.add_argument('--folder', type=str, default=config.get('folder', '.'), help='Folder to save the results.')
    parser.add_argument('--step', type=int, default=config.get("step",1), help='Step size for sliding window.')
    parser.add_argument('--kmeans_init', type=bool, default=config.get("kmeans_init",True), help='Step size for sliding window.')
    args = parser.parse_args()

    return args

def train(index, args):
    print(args)
    folder = args.folder
    os.makedirs(folder, exist_ok=True)
    # Load the main data (no UA column)
    model_data = pd.read_csv("./data/processed_features_and_clinical_data.csv")
    ID_label = pd.read_csv("./data/case_control.csv")

    model_data['row_number'] = model_data.index
    ID_label.rename(columns={"Mother_MRN_obfuscated": 'Mother_Obfus_MRN'}, inplace=True)
    # Load the UA list as JSON
    with open("./data/ua_list.json", "r") as f:
        ua_list = json.load(f)
    linked_data = pd.merge(model_data, ID_label, on='Mother_Obfus_MRN', how="inner")
    linked_data = linked_data.drop_duplicates(subset='Mother_Obfus_MRN', keep="first")

    time_series = []
    for row_number in linked_data['row_number']:
        time_series.append(ua_list[row_number])

    mean_list = [take_mean(ts, window_size=mean_window_size) for ts in time_series]


    # train test split
    labels = linked_data['Case_Control'].to_list()
    ids = linked_data['Mother_Obfus_MRN'].to_list()
    label_id = list(zip(labels, ids))
    train_list, test_list, train_label_id, test_label_id = train_test_split(mean_list,label_id, test_size=0.2, shuffle=True, random_state=42)
    # train_list, val_list, train_label_id, val_label_id = train_test_split(train_list,train_label_id, test_size=0.2, shuffle=True, random_state=42)
    train_label = [train_label_id[i][0] for i in range(len(train_label_id))]
    # val_label = [val_label_id[i][0] for i in range(len(val_label_id))]
    test_label = [test_label_id[i][0] for i in range(len(test_label_id))]
    train_id = [train_label_id[i][1] for i in range(len(train_label_id))]
    # val_id = [val_label_id[i][1] for i in range(len(val_label_id))]
    test_id = [test_label_id[i][1] for i in range(len(test_label_id))]

    x_train, label_train, id_train = segmentation(train_list, train_label, train_id)
    # x_val, label_val, id_val = segmentation(val_list, val_label, val_id)
    x_test, label_test, id_test = segmentation(test_list, test_label, test_id)
    y_train = np.unique(label_train, return_inverse=True)[1]
    # y_val = np.unique(label_val, return_inverse=True)[1]
    y_test = np.unique(label_test, return_inverse=True)[1]


    n_ts, n_channels, len_ts = x_train.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_train))
    num_shapelets = int(args.num_shapelets_ratio * x_train.shape[2])
    if num_shapelets % args.nhead != 0:
        num_shapelets = int(num_shapelets // args.nhead) * args.nhead
    shapelets_size_and_len = {
        int(args.size_ratio*x_train.shape[2]): num_shapelets, 
        # int(0.25*x_train.shape[2]): int(0.1*x_train.shape[2])    
    }

    # set n head as the division of number of shapelets
    model = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len, 
                              seq_len=len_ts,
                            in_channels = n_channels,
                            step=args.step,
                            num_classes = num_classes,
                            loss_func = loss_func,
                            to_cuda = True,
                            verbose = 1,
                            dist_measure = args.dist_measure, 
                            nhead=args.nhead,
                            num_layers=args.num_layers
                            )
    if (args.kmeans_init):
        for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len.items()):
            weights_block = get_weights_via_kmeans(x_train, shapelets_size, num_shapelets)
            model.set_shapelet_weights_of_block(i, weights_block)
    optimizer = optim.Adam(model.model.parameters(), lr=args.lr, weight_decay=args.wd, eps=args.epsilon)
    model.set_optimizer(optimizer)

    loss = model.fit(x_train, y_train, epochs=args.num_epochs, batch_size=args.batch_size, shuffle=True, drop_last=False)



    # fig = plt.figure()
    # plt.plot(loss, color='black')
    # plt.title("Loss over training steps")
    # fig.savefig("./"+folder+f'/loss_transformer_{index}.png')
    # plt.close()

    acc = eval_accuracy(model, x_test, y_test)

    shapelets = model.get_shapelets()

    # for i in range(2):
    #     plt.plot(shapelets[i, 0])
    # shapelet_transform = model.transform(x_train)

    # dist_s1 = shapelet_transform[:, 0]
    # dist_s2 = shapelet_transform[:, 1]

    # weights, biases = model.get_weights_linear_layer()

    # fig = plt.figure(facecolor='white')
    # fig.set_size_inches(20, 8)
    # gs = fig.add_gridspec(12, 8)
    # fig_ax1 = fig.add_subplot(gs[0:3, :4])
    # fig_ax1.set_title("First learned shapelet plotted (in red) on top of its 10 best matching time series.")
    # for i in np.argsort(dist_s1)[:10]:
    #     fig_ax1.plot(x_train[i, 0], color='black', alpha=0.5)
    #     _, pos = torch_dist_ts_shapelet(x_train[i], shapelets[0])
    #     fig_ax1.plot(lead_pad_shapelet(shapelets[0, 0], pos), color='#F03613', alpha=0.5)

    # fig_ax2 = fig.add_subplot(gs[0:3, 4:])
    # fig_ax2.set_title("Second learned shapelet plotted (in red) on top of its 10 best matching time series.")
    # for i in np.argsort(dist_s2)[:10]:
    #     fig_ax2.plot(x_train[i, 0], color='black', alpha=0.5)
    #     _, pos = torch_dist_ts_shapelet(x_train[i], shapelets[1])
    #     fig_ax2.plot(lead_pad_shapelet(shapelets[1, 0], pos), color='#F03613', alpha=0.5)


    # fig_ax3 = fig.add_subplot(gs[4:, :])
    # fig_ax3.set_title("The decision boundaries learned by the model to separate the two classes.")
    # color = {0: '#F03613', 1: '#7BD4CC'}
    # fig_ax3.scatter(dist_s1, dist_s2, color=[color[l] for l in y_train])

    # # viridis = plt.get_cmap('viridis', 4)
    # # # Create a meshgrid of the decision boundaries
    # # xmin = np.min(shapelet_transform[:, 0]) - 0.1
    # # xmax = np.max(shapelet_transform[:, 0]) + 0.1
    # # ymin = np.min(shapelet_transform[:, 1]) - 0.1
    # # ymax = np.max(shapelet_transform[:, 1]) + 0.1
    # # xx, yy = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin)/200),
    # #                         np.arange(ymin, ymax, (ymax - ymin)/200))

    # fig.savefig("./" + folder + f"/distribution_trans_{index}.png")
    
    return acc
def save_results_to_csv(results, filename="results.csv"):
        keys = results[0].keys()
        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

if __name__ == "__main__":
    results = []
    avg_results = []
    
    yaml_files = glob.glob(os.path.join(root, "yaml_configs_new/*.yaml"))
    for i, config_path in enumerate(yaml_files):
        acc_list = []
        time_list = []
        args = parse_args(config_path)
        
        start = time.time()
        acc = train(index=i, args=args)
        end = time.time()
        elapsed = end - start
        result = {     
            "index": i,
            "accuracy": acc,
            "elapsed_time": elapsed
        }
        for key, value in vars(args).items():
            result[key] = value
        results.append(result)

        print(f"Total elapsed time: {elapsed:.2f} seconds")
        print("------------------------------------------")


        # for j in range(10):
        #     start = time.time()
        #     acc = train(index=i, argas=args)
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
        #     acc_list.append(acc)
        #     time_list.append(elapsed)

        #     print(f"Total elapsed time: {elapsed:.2f} seconds")
        #     print("------------------------------------------")
        # avg_result = {
        #     "accuracy": np.mean(acc_list),
        #     "elapsed_time": np.mean(time_list)
        # }
        # for key, value in vars(args).items():
        #     avg_result[key] = value
        # avg_results.append(avg_result)

    save_results_to_csv(results, filename="results_preterm2.csv")
    # save_results_to_csv(avg_results, filename="results_preterm.csv")