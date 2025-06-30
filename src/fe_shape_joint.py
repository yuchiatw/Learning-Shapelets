import tsfel
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing

import torch
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
import warnings
import sys
import os

from src.learning_shapelets_sliding_window import ShapeletsDistBlocks

def feature_extraction_selection(X_train, threshold=0):
    corr_features, X_train = tsfel.correlated_features(X_train, drop_correlated=True)
    selector = VarianceThreshold(threshold=threshold)
    X_train = selector.fit_transform(X_train)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, corr_features, selector, scaler

def extraction_pipeline(Feature, corr_features, selector, scaler):
    Feature.drop(corr_features, axis=1, inplace=True)
    Feature = selector.transform(Feature)
    nFeature = scaler.transform(Feature)
    return nFeature

class JointModel(nn.Module):
    """
    From the input sequence, out a sequence of selected features.
    """
    def __init__(
        self, 
        shapelets_size_and_len, 
        seq_len, 
        num_features, 
        mode = 'fusion',
        window_size = 30, 
        in_channels = 1, 
        step = 1, 
        num_classes = 2, 
        dist_measure = 'euclidean',
        nhead = 2, 
        num_layers = 4, 
        batch_first = True,
        to_cuda=True
    ):
        super(JointModel, self).__init__()
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.num_features = num_features
        self.d_model = self.num_shapelets
        self.mode = mode
        print(seq_len, window_size, step)
        self.transform_seq_len = int((seq_len - window_size)/step+1)
        
        
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=in_channels,
                                                    step=step, window_size=window_size,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure, to_cuda=to_cuda)
        
        self.proj_1 = nn.Linear(self.num_shapelets, self.d_model)
        self.proj_2 = nn.Linear(self.num_features, self.d_model)
        self.fusion_weights = nn.Linear(self.d_model * 2, 2)
        
        self.positional_emb = nn.Embedding(self.transform_seq_len, self.d_model)
        print(self.d_model, nhead)
        encoder_layers = TransformerEncoderLayer(d_model=self.d_model, nhead = nhead, batch_first=batch_first)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.concat_layer = nn.Linear(self.d_model*2, self.d_model)    
        # Optional fusion network
        self.linear = nn.Linear(self.d_model, num_classes)
        self.max = 0
        if self.to_cuda:
            self.cuda()
    def forward(self, x, feature_sequence, update_max=False):
        
        x, y = self.shapelets_blocks(x)
        y = torch.squeeze(y, 1)
        # if update_max:
        #     max_value = torch.max(y)
        #     self.max = torch.max(torch.tensor([max_value, self.max]))
        # y = 1 - y / self.max
        # y = F.normalize(y, p=2, dim=-1)
        y_min = y.min(dim=-1, keepdim=True)[0]
        y_max = y.max(dim=-1, keepdim=True)[0]
        y = (y - y_min) / (y_max - y_min + 1e-8)
        y = 1 - y
        
        if self.mode == 'fusion':
            # Apply dynamic weighting
            f1_proj = self.proj_1(y)
            f2_proj = self.proj_2(feature_sequence)
            cat_features = torch.cat([f1_proj, f2_proj], dim=-1) # (batch, time_steps, d_model*2)
            fusion_scores = F.softmax(self.fusion_weights(cat_features), dim=-1)
            F1_weighted = fusion_scores[..., 0].unsqueeze(-1) * f1_proj  # (batch, time_steps, d_model)
            F2_weighted = fusion_scores[..., 1].unsqueeze(-1) * f2_proj  # (batch, time_steps, d_model)
            joint_output = F1_weighted + F2_weighted  # (batch, time_steps, d_model)
        else:
            f1_proj = self.proj_1(y)
            f2_proj = self.proj_2(feature_sequence)
            cat_features = torch.cat([f1_proj, f2_proj], dim=-1)
            joint_output = self.concat_layer(cat_features)
        
        batch_size, _, _ = y.shape
        pos_indices = torch.arange(self.transform_seq_len, device=x.device).unsqueeze(0).expand(batch_size, self.transform_seq_len)
        pos_emb = self.positional_emb(pos_indices)
        transformer_out = self.transformer_encoder(joint_output + pos_emb)
        # final_out = self.linear(transformer_out[:, -1, :])
        final_out = self.linear(transformer_out.mean(dim=1))
        return final_out
class ShapeletsDistanceLoss(nn.Module):
    """
    Calculates the cosine similarity of a bunch of shapelets to a data set and performs global max-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, dist_measure='euclidean', k=6):
        super(ShapeletsDistanceLoss, self).__init__()
        if not dist_measure == 'euclidean' and not dist_measure == 'cosine':
            raise ValueError("Parameter 'dist_measure' must be either of 'euclidean' or 'cosine'.")
        if not isinstance(k, int):
            raise ValueError("Parameter 'k' must be an integer.")
        self.dist_measure = dist_measure
        self.k = k

    def forward(self, x):
        """
        Calculate the loss as the average distance to the top k best-matching time series.
        @param x: the shapelet transform
        @type x: tensor(float) of shape (batch_size, n_shapelets)
        @return: the computed loss
        @rtype: float
        """
        y_top, y_topi = torch.topk(x.clamp(1e-8), self.k, largest=False if self.dist_measure == 'euclidean' else True,
                                   sorted=False, dim=0)
        # avoid compiler warning
        y_loss = None
        if self.dist_measure == 'euclidean':
            y_loss = torch.mean(y_top)
        elif self.dist_measure == 'cosine':
            y_loss = torch.mean(1 - y_top)
        return y_loss
class ShapeletsSimilarityLoss(nn.Module):
    """
    Calculates the cosine similarity of each block of shapelets and averages over the blocks.
    ----------
    """
    def __init__(self):
        super(ShapeletsSimilarityLoss, self).__init__()

    def cosine_distance(self, x1, x2=None, eps=1e-8):
        """
        Calculate the cosine similarity between all pairs of x1 and x2. x2 can be left zero, in case the similarity
        between solely all pairs in x1 shall be computed.
        @param x1: the first set of input vectors
        @type x1: tensor(float)
        @param x2: the second set of input vectors
        @type x2: tensor(float)
        @param eps: add small value to avoid division by zero.
        @type eps: float
        @return: a distance matrix containing the cosine similarities
        @type: tensor(float)
        """
        x2 = x1 if x2 is None else x2
        # unfold time series to emulate sliding window
        x1 = x1.unfold(2, x2.shape[2], 1).contiguous()
        x1 = x1.transpose(0, 1)
        # normalize with l2 norm
        x1 = x1 / x1.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        x2 = x2 / x2.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)

        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x1 = torch.matmul(x1, x2.transpose(1, 2))
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        # and average over dims to keep range between 0 and 1
        n_dims = x1.shape[1]
        x1 = torch.sum(x1, dim=1) / n_dims
        return x1

    def forward(self, shapelet_blocks):
        """
        Calculate the loss as the sum of the averaged cosine similarity of the shapelets in between each block.
        @param shapelet_blocks: a list of the weights (as torch parameters) of the shapelet blocks
        @type shapelet_blocks: list of torch.parameter(tensor(float))
        @return: the computed loss
        @rtype: float
        """
        losses = 0.
        for block in shapelet_blocks:
            shapelets = block[1]
            shapelets.retain_grad()
            sim = self.cosine_distance(shapelets, shapelets)
            losses += torch.mean(sim)
        return losses

class JointTraining:
    def __init__(self, shapelets_size_and_len, 
                 seq_len, 
                 num_features, 
                 loss_func,
                 mode = 'fusion',
                 window_size = 30, 
                 in_channels=1, step=1, 
                 num_classes=2, dist_measure='euclidean',
                 nhead = 2, num_layers = 4, 
                 verbose = 1, k=0, l1=0.0, l2=0.0, 
                 batch_first = True,
                 to_cuda=True):
        self.model = JointModel(
            shapelets_size_and_len = shapelets_size_and_len, 
            seq_len = seq_len, 
            num_features = num_features, 
            mode = mode,
            window_size = window_size, 
            in_channels=in_channels, 
            step=step, 
            num_classes=num_classes, 
            dist_measure=dist_measure,
            nhead = nhead, 
            num_layers = num_layers, 
            batch_first = batch_first,
            to_cuda=to_cuda
        )
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.loss_func = loss_func
        self.verbose = verbose
        self.optimizer = None
        self.k = k
        self.l1 = l1
        self.l2 = l2
        self.loss_dist = ShapeletsDistanceLoss(dist_measure=dist_measure, k=k)
        self.loss_sim_block = ShapeletsSimilarityLoss()
    def set_optimizer(self, optimizer):
        """
        Set an optimizer for training.
        @param optimizer: a PyTorch optimizer: https://pytorch.org/docs/stable/optim.html
        @type optimizer: torch.optim
        @return:
        @rtype: None
        """
        self.optimizer = optimizer
    def set_shapelet_weights(self, weights):
        """
        Set the weights of all shapelets. The shapelet weights are expected to be ordered ascending according to the
        length of the shapelets. The values in the matrix for shapelets of smaller length than the maximum
        length are just ignored.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (in_channels, num_total_shapelets, shapelets_size_max)
        @return:
        @rtype: None
        """
        start = 0
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            end = start + num_shapelets
            self.model.shapelets_blocks.set_shapelet_weights_of_block(i, weights[start:end, :, :shapelets_size])
            start = end

        if self.optimizer is not None:
            warnings.warn("Updating the model parameters requires to reinitialize the optimizer. Please reinitialize"
                          " the optimizer via set_optimizer(optim)")
    def set_shapelet_weights_of_block(self, i, weights):
        """
        Set the weights of shapelet block i.
        @param i: The index of the shapelet block
        @type i: int
        @param weights: the weights for the shapelets of block i
        @type weights: array-like(float) of shape (in_channels, num_shapelets, shapelets_size)
        @return:
        @rtype: None
        """
        self.model.shapelets_blocks.set_shapelet_weights_of_block(i, weights)
        if self.optimizer is not None:
            warnings.warn("Updating the model parameters requires to reinitialize the optimizer. Please reinitialize"
                          " the optimizer via set_optimizer(optim)")

    def get_shapelets(self):
        return self.model.shapelets_blocks.get_shapelets().clone().cpu().detach().numpy()
    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        if self.to_cuda:
            X = X.cuda()

        with torch.no_grad():
            shapelet_transform, _ = self.model.shapelets_blocks(X)
        return shapelet_transform.squeeze().cpu().detach().numpy()
    def update(self, x, fe, y, update_max='false'):
        """
        Performs one gradient update step for the batch of time series and corresponding labels y.
        @param x: the batch of time series
        @type x: array-like(float) of shape (n_batch, in_channels, len_ts)
        @param y: the labels of x
        @type y: array-like(long) of shape (n_batch)
        @return: the loss for the batch
        @rtype: float
        """
        yhat = self.model(x, fe, update_max)
        loss = self.loss_func(yhat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    def loss_sim(self):
        """
        Get the weights of each shapelet block and calculate the cosine distance between the
        shapelets inside each block and return the summed distances as their similarity loss.
        @return: the shapelet similarity loss for the batch
        @rtype: float
        """
        blocks = [params for params in self.model.named_parameters() if 'shapelets_blocks' in params[0]]
        loss = self.loss_sim_block(blocks)
        return loss
    def update_regularized(self, x, fe, y):
        """
        Performs one gradient update step for the batch of time series and corresponding labels y using the
        loss L_r.
        @param x: the batch of time series
        @type x: array-like(float) of shape (n_batch, in_channels, len_ts)
        @param y: the labels of x
        @type y: array-like(long) of shape (n_batch)
        @return: the three losses cross-entropy, shapelet distance, shapelet similarity for the batch
        @rtype: Tuple of float
        """
        # get cross entropy loss and compute gradients
        y_hat = self.model(x, fe)
        loss_ce = self.loss_func(y_hat, y)
        loss_ce.backward(retain_graph=True)

        # get shapelet distance loss and compute gradients
        dists_mat = self.model(x, 'dists')
        loss_dist = self.loss_dist(dists_mat) * self.l1
        loss_dist.backward(retain_graph=True)

        if self.l2 > 0.0:
            # get shapelet similarity loss and compute gradients
            loss_sim = self.loss_sim() * self.l2
            loss_sim.backward(retain_graph=True)

        # perform gradient upgrade step
        self.optimizer.step()
        self.optimizer.zero_grad()

        return (loss_ce.item(), loss_dist.item(), loss_sim.item()) if self.l2 > 0.0 else (
        loss_ce.item(), loss_dist.item())

    def fit(
        self,
        X, FE, Y, 
        X_val = None,
        FE_val = None, 
        Y_val = None,
        epochs=1, 
        batch_size=256, 
        shuffle=False, 
        drop_last=False, 
        patience=10, 
        min_delta=1e-6,
        model_path="./best_model.pth",
    ):
        
        if self.optimizer == None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")
        
        if not isinstance(X, torch.Tensor):
            X = tensor(X, dtype=torch.float).contiguous()
        if not isinstance(FE, torch.Tensor):
            FE = tensor(FE, dtype=torch.float).contiguous()
        if not isinstance(Y, torch.Tensor):
            Y = tensor(Y, dtype=torch.long).contiguous()
        if self.to_cuda:
            X = X.cuda()
            FE = FE.cuda()
            Y = Y.cuda()
        
        val_dataset = None
        val_loader = None
        if X_val is not None and Y_val is not None and FE_val is not None:
            if not isinstance(X_val, torch.Tensor):
                X_val = tensor(X_val, dtype=torch.float).contiguous()
            if not isinstance(FE_val, torch.Tensor):
                FE_val = tensor(FE_val, dtype=torch.float).contiguous()
            if not isinstance(Y_val, torch.Tensor):
                Y_val = tensor(Y_val, dtype=torch.long).contiguous()
            if self.to_cuda:
                X_val = X_val.cuda()
                FE_val = FE_val.cuda()
                Y_val = Y_val.cuda()
            val_dataset = TensorDataset(X_val, FE_val, Y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            
        

        train_dataset = TensorDataset(X, FE, Y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) 
        
        
        progress_bar = tqdm(range(epochs), disable=False)
        self.model.train()
        loss_list = []
        val_loss = []
        best_val_loss = float('inf')
        epochs_no_improve=0
        early_stop = False
        for epoch in progress_bar:
            batch_loss = []
            for j, (x, fe, y) in enumerate(train_loader):
                update_max = False
                if epoch == 0:
                    update_max = True
                loss = self.update(x, fe, y, update_max=update_max)
                batch_loss.append(loss)
            loss_list.append(sum(batch_loss) / len(batch_loss))
            batch_loss = []
            if val_loader is not None:
                self.model.eval()
                batch_loss = []
                with torch.no_grad():
                    for j, (x, fe, y) in enumerate(val_loader):
                        yhat = self.model(x, fe)
                        loss = self.loss_func(yhat, y)
                        batch_loss.append(loss.item())
                val_loss_epoch = sum(batch_loss) / len(batch_loss)
                val_loss.append(val_loss_epoch)
                
                # Early stopping
                if val_loss_epoch < best_val_loss:
                    best_val_loss = val_loss_epoch
                    epochs_no_improve = 0
                    torch.save(self.model.state_dict(), model_path)
                else:
                    epochs_no_improve += 1
                
                # if epochs_no_improve >= patience:
                #     print(f"Early stopping at epoch {epoch}")
                #     break
                
                self.model.train()
            
        return loss_list, best_val_loss   
            
    def predict(
        self, 
        X, FE,
        batch_size=256, 
        to_cuda=True
    ):
        if not isinstance(X, torch.Tensor):
            X = tensor(X, dtype=torch.float32).contiguous()
        if not isinstance(FE, torch.Tensor):
            FE = tensor(FE, dtype=torch.float32).contiguous()
        if to_cuda:
            X = X.cuda()
            FE = FE.cuda()
        eval_dataset = TensorDataset(X, FE)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        self.model.eval()
        result=None
        with torch.no_grad():
            for (x, fe) in eval_loader:
                y_hat = self.model(x, fe)
                y_hat = y_hat.cpu().detach()
                result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
        
        return result
    def load_model(self, model_path='./model/best_model.pth'):
        self.model.load_state_dict((torch.load(model_path, weights_only=True)))


if __name__ == '__main__':
    
    data = np.load('./data/ECG200.npz')
    x_train = data['X_train'].transpose(0, 2, 1)
    x_val = data['X_val'].transpose(0, 2, 1)
    x_test = data['X_test'].transpose(0, 2, 1)
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    num_classes = len(set(y_train))
    num_train, len_ts, in_channels = x_train.shape
    num_val = x_val.shape[0]
    num_test = x_test.shape[0]
    cfg_file = tsfel.get_features_by_domain()
    X_train = tsfel.time_series_features_extractor(cfg_file, x_train)
    window_size = 20
    window_step = 5
    
    x_train_split = sliding_window_view(x_train, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
    x_val_split = sliding_window_view(x_val, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
    x_test_split = sliding_window_view(x_test, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
    
    print(x_train_split.shape)
    print(x_val_split.shape)
    print(x_test_split.shape)
    num_windows = x_test_split.shape[1]
    x_train_split = x_train_split.reshape(num_train * num_windows, window_size, in_channels)
    x_val_split = x_val_split.reshape(num_val * num_windows, window_size, in_channels)
    x_test_split = x_test_split.reshape(num_test * num_windows, window_size, in_channels)
    
    X_train_split = tsfel.time_series_features_extractor(cfg_file, x_train_split)
    X_val_split = tsfel.time_series_features_extractor(cfg_file, x_val_split)
    X_test_split = tsfel.time_series_features_extractor(cfg_file, x_test_split)
    X_train_split_filtered, corr_features, selector, scaler = feature_extraction_selection(X_train_split)
    X_val_split_filtered = extraction_pipeline(X_val_split, corr_features, selector, scaler)
    X_test_split_filtered = extraction_pipeline(X_test_split, corr_features, selector, scaler)
    
    
    X_train_split_filtered = X_train_split_filtered.reshape(num_train, num_windows, -1)
    X_val_split_filtered = X_val_split_filtered.reshape(num_val, num_windows, -1)
    X_test_split_filtered = X_test_split_filtered.reshape(num_test, num_windows, -1)
    num_features = X_train_split_filtered.shape[-1]
    shapelets_size_and_len = {10: 2, 15: 2}
    
    model_config = {
        'epochs': 200, 
        'batch_size': 32, 
        'model_path': './model/best_model.pth',
        'step': 1,
        'lr': 1e-3, 
        'wd': 1e-4, 
        'epsilon': 1e-7,
        'l2': 1e-4
    }
    loss_func = nn.CrossEntropyLoss()
    model = JointTraining(
        shapelets_size_and_len=shapelets_size_and_len,
        seq_len=len_ts, 
        in_channels=in_channels, 
        loss_func = loss_func, 
        mode = 'concat', 
        num_features=num_features, 
        window_size=window_size, 
        step=window_step,
        num_classes=num_classes, 
        to_cuda = True
    )
    optimizer = optim.Adam(
        model.model.parameters(), 
        lr=model_config['lr'], 
        weight_decay=model_config['wd'], 
        eps=model_config['epsilon']
    )
    model.set_optimizer(optimizer=optimizer)
    loss_list, val_loss = model.fit(
        data['X_train'], X_train_split_filtered, y_train,
        X_val=data['X_val'], FE_val = X_val_split_filtered, Y_val=y_val,
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size'],
        shuffle=True, 
        model_path=model_config['model_path']
    )
    model.load_model(model_config['model_path'])
    y_pred = model.predict(data['X_test'], X_test_split_filtered)
    
    # results = eval_results(y_test, y_pred)