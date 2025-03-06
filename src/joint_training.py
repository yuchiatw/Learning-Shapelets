import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())
import warnings

from src.learning_shapelets_sliding_window import LearningShapeletsModel
from src.vanilla_transformer import TimeSeriesTransformer
import torch
from torch import tensor
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

class MultiBranchModel(nn.Module):
    def __init__(
        self, shapelets_size_and_len, seq_len, 
        window_size = 30, in_channels=1, step=1, 
        num_classes=2, dist_measure='euclidean',
        nhead = 2, num_layers = 4, d_model = 4,
        to_cuda=True
    ):
        super(MultiBranchModel, self).__init__()
        
        # Create branches with focused feature extraction
        self.to_cuda = to_cuda
        self.LearningShapelets = LearningShapeletsModel(
            shapelets_size_and_len=shapelets_size_and_len,
            seq_len=seq_len, 
            window_size=window_size,
            in_channels=in_channels,
            step=step,
            num_classes=num_classes,
            num_layers=num_layers,
            nhead=nhead,
            dist_measure=dist_measure,
            to_cuda=to_cuda
        )
        self.vanilla = TimeSeriesTransformer(
            seq_len=seq_len,
            num_classes=num_classes,
            channels=in_channels,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            to_cuda=to_cuda
        )
        
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.randn(2))  # Two branches
    
        # Optional fusion network
        self.fusion_layer = nn.Linear(num_classes, num_classes)
        if self.to_cuda:
            self.cuda()

    def forward(self, x):
        # Extract features from both branches
        shapelets_out = self.LearningShapelets(x)  # Output: (batch, num_classes)
        transformer_out = self.vanilla(x)          # Output: (batch, num_classes)
        
        # # Compute dynamic weights (softmax for normalization)
        fusion_weights = F.softmax(self.fusion_weights, dim=0)  # (2,)
        
        # # Apply weighted fusion
        fused_output = (fusion_weights[0] * shapelets_out) + (fusion_weights[1] * transformer_out)
        
        # # Optional refinement via fusion network
        fused_output = self.fusion_layer(fused_output)

        return fused_output
class MultiBranch:
    def __init__(self, shapelets_size_and_len, seq_len, 
        loss_func,
        window_size = 30, in_channels=1, step=1, 
        num_classes=2, dist_measure='euclidean',
        nhead = 2, num_layers = 4, d_model = 4,
        verbose = 0,
        to_cuda=True
    ):
        self.model = MultiBranchModel(
            shapelets_size_and_len=shapelets_size_and_len,
            seq_len=seq_len,
            window_size=window_size,
            in_channels=in_channels,
            step=step,
            num_classes=num_classes,
            num_layers=num_layers,
            nhead=nhead, d_model=d_model,
            dist_measure=dist_measure,
            to_cuda=to_cuda
        )
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.loss_func = loss_func
        self.verbose = verbose
        self.optimizer = None
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
        self.model.LearningShapelets.set_shapelet_weights(weights)
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
        self.model.LearningShapelets.set_shapelet_weights_of_block(i, weights)
        if self.optimizer is not None:
            warnings.warn("Updating the model parameters requires to reinitialize the optimizer. Please reinitialize"
                          " the optimizer via set_optimizer(optim)")


    def fit(
        self,
        X, Y, 
        X_val = None,
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
        if not isinstance(Y, torch.Tensor):
            Y = tensor(Y, dtype=torch.long).contiguous()
        if self.to_cuda:
            X = X.cuda()
            Y = Y.cuda()
        
        val_dataset = None
        val_loader = None
        if X_val is not None and Y_val is not None:
            if not isinstance(X_val, torch.Tensor):
                X_val = tensor(X_val, dtype=torch.float32).contiguous()
            if not isinstance(Y_val, torch.Tensor):
                Y_val = tensor(Y_val, dtype=torch.long).contiguous()
            if self.to_cuda:
                X_val = X_val.cuda()
                Y_val = Y_val.cuda()
            val_dataset = TensorDataset(X_val, Y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            
        

        train_dataset = TensorDataset(X, Y)
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
            for j, (x, y) in enumerate(train_loader):
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_loss.append(loss.item())
            loss_list.append(sum(batch_loss) / len(batch_loss))
            batch_loss = []
            if val_loader is not None:
                self.model.eval()
                batch_loss = []
                with torch.no_grad():
                    for j, (x, y) in enumerate(val_loader):
                        yhat = self.model(x)
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
        
        self.model.eval()
        result=None
        with torch.no_grad():
            for x in eval_loader:
                y_hat = self.model(x[0])
                y_hat = y_hat.cpu().detach()
                result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
        
        return result
    def load_model(self, model_path='./model/best_model.pth'):
        self.model.load_state_dict((torch.load(model_path, weights_only=True)))
        
# --------------- Usage Example --------------------
if __name__ == "__main__":
    # Hyperparameters
    SEQ_LEN = 50    # e.g., each time series has length=50
    CHANNELS = 1    # univariate
    D_MODEL = 64
    NHEAD = 2
    NUM_LAYERS = 2
    NUM_CLASSES = 2 # example number of classes
    BATCH_SIZE = 16
    x = torch.randn(BATCH_SIZE, CHANNELS, SEQ_LEN)
    shapelets_size_and_len = {10: 2, 15: 2}
    model = MultiBranchModel(
        shapelets_size_and_len=shapelets_size_and_len,
        seq_len=SEQ_LEN,
        in_channels=CHANNELS,
        window_size=15,
        step=1,
        num_classes=NUM_CLASSES,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        d_model=4, 
        to_cuda=False
        
    )
    # Forward pass
    logits = model(x)
    print("Logits shape:", logits.shape)  # [16, 2]

    # Define a loss, optimizer
    labels = torch.randint(high=NUM_CLASSES, size=(BATCH_SIZE,))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())
