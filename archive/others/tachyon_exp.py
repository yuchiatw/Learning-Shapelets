import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocessing import normalize_data
import argparse

parser = argparse.ArgumentParser(description='Tachyon Shapelet Learning Experiment')
parser.add_argument('--feature', type=str, default='part_max_used',
                  help='Feature to use for shapelet learning')
Args = parser.parse_args()
feature = Args.feature
print(f"Using feature: {feature}")
data = pd.read_csv('tachyon/far_data_2024-02-21_clustered.csv')

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['nodeId'].unique().shape
from utils.preprocessing import normalize_data

time_series = []
for node_id in data['nodeId'].unique():
    ts = data[data['nodeId'] == node_id].reset_index(drop=True)
    time_series.append(ts)

def segment_time_series(series, segment_length = 600, step = 300):
    n = len(series)
    # Number of complete segments
    num_segments = n // step
    # Segment the series and assign the label to each segment
    segments = [
        series[i * step:i * step + segment_length].reset_index(drop=False)
        for i in range(num_segments) if i * step + segment_length < n
    ]
    return segments


segments = []
for i in range(len(time_series)):
    # Segment the time series for each node
    segments_i = segment_time_series(time_series[i])
    segments.extend(segments_i)

# segemented time series and respective labels
segment_cpu = []
labels = []
for segment in segments:
    # Calculate the mean CPU usage for each segment
    cpu = segment[feature]
    label = segment['Cluster'][0]
    labels.append(label)
    segment_cpu.append(cpu)
    
print(labels[:10])
print(len(labels))

normalize_segment_cpu, scaler = normalize_data(np.array(segment_cpu))

segment_array = normalize_segment_cpu.reshape(len(normalize_segment_cpu), 1, len(normalize_segment_cpu[0]))
labels = np.array(labels)

train_data, test_data, train_labels, test_labels = train_test_split(
    segment_array, labels, test_size=0.2, random_state=42, stratify=labels
)

from src.learning_shapelets import LearningShapelets
from tslearn.clustering import TimeSeriesKMeans
import random
import torch
from torch import nn, optim

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

n_ts, n_channels, len_ts = train_data.shape
loss_func = nn.CrossEntropyLoss()
num_classes = len(set(train_labels))
shapelets_size_and_len = {50: 10, 25: 10, 10: 10}
dist_measure = "euclidean"
lr = 1e-2
wd = 1e-3
epsilon = 1e-7
print(train_data.shape, test_data.shape)
learning_shapelets = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len,
                                       in_channels=n_channels,
                                       num_classes=num_classes,
                                       loss_func=loss_func,
                                       to_cuda=True,
                                       verbose=1,
                                       dist_measure=dist_measure)
optimizer = optim.Adam(learning_shapelets.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
learning_shapelets.set_optimizer(optimizer)
losses = learning_shapelets.fit(train_data, train_labels, epochs=3000, batch_size=256, shuffle=False, drop_last=False)

import matplotlib.pyplot as plt
plt.plot(losses, color="black")
plt.show()


def eval_accuracy(model, X, Y):
    predictions = model.predict(X)
    if len(predictions.shape) == 2:
        predictions = predictions.argmax(axis=1)
    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")
    
eval_accuracy(learning_shapelets, test_data, test_labels)
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

shapelets = learning_shapelets.get_shapelets()
shapelet_transform = learning_shapelets.transform(train_data)
dist_s1 = shapelet_transform[:, 0]
dist_s2 = shapelet_transform[:, 1]

fig = plt.figure()
gs = fig.add_gridspec(12, 8)
fig_ax1 = fig.add_subplot(gs[0:3, :4])
fig_ax1.set_title("First learned shapelet")
for i in np.argsort(dist_s1)[:10]:
    fig_ax1.plot(train_data[i, 0], color='black', alpha=0.5)
    _, pos = torch_dist_ts_shapelet(train_data[i, 0:50], shapelets[0])
    fig_ax1.plot(lead_pad_shapelet(shapelets[0, 0], pos), color='#F03613', alpha=0.5)

fig_ax2 = fig.add_subplot(gs[0:3, 4:])
fig_ax2.set_title("Second learned shapelet")
for i in np.argsort(dist_s2)[:10]:
    fig_ax2.plot(train_data[i, 0], color='black', alpha=0.5)
    _, pos = torch_dist_ts_shapelet(train_data[i,0:50], shapelets[1])
    fig_ax2.plot(lead_pad_shapelet(shapelets[1, 0], pos), color='#F03613', alpha=0.5) 
plt.title(f"Shapelets for {feature}")
fig.savefig(f'tachyon/tachyon_shapelets_{feature}.png')
