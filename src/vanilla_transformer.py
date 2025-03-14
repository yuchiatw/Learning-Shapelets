import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

class TimeSeriesTransformer(nn.Module):
    """
    Implement Vanilla Transformer
    --------
    Args:
        seq_len    : Maximum sequence length expected.
        d_model    : Dimensionality of the Transformer embeddings.
        nhead      : Number of attention heads.
        num_layers : Number of Transformer encoder layers.
        num_classes: Number of target classes.
        channels   : Number of channels in the input (C).
        batch_first: If True, the transformer expects [B, L, E], else [L, B, E].
    """
    def __init__(
        self,
        seq_len: int,
        num_classes: int,
        channels: int = 1,   # Number of input channels/features
        d_model: int = 4,
        nhead: int = 2,
        num_layers: int = 2,
        batch_first: bool = True, 
        to_cuda: bool = True,
    ):
        
        super().__init__()
        self.to_cuda = to_cuda
        self.seq_len = seq_len
        self.d_model = d_model
        self.batch_first = batch_first
        
        # 1) Linear projection from input channels to d_model
        self.input_projection = nn.Linear(channels, d_model)
        
        # 2) Learned positional embedding
        self.positional_emb = nn.Embedding(seq_len, d_model)
        
        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=batch_first,  # PyTorch 2.0+ 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4) Classification head
        self.fc = nn.Linear(d_model, num_classes)
        
        if self.to_cuda:
            self.cuda()

    def forward(self, x):
        """
        x shape: [batch_size, channels, seq_len]
        Returns: [batch_size, num_classes]
        """
        # (A) Permute to [batch_size, seq_len, channels]
        #     so each time step is a row in the sequence dimension
        x = x.permute(0, 2, 1)  # => [B, L, C]
        
        batch_size, seq_len, _ = x.shape
        
        # (B) Project to d_model: [B, L, C] => [B, L, d_model]
        x = self.input_projection(x)
        
        # (C) Create position indices: shape => [B, L]
        pos_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # (D) Get positional embeddings: [B, L, d_model]
        pos_emb = self.positional_emb(pos_indices)
        
        # (E) Add positional embeddings to input
        x = x + pos_emb
        
        # (F) Pass through Transformer encoder
        x = self.transformer_encoder(x)  # => [B, L, d_model] (with batch_first=True)
        
        # (G) Pool over sequence dimension: mean pool as an example
        x = x.mean(dim=1)  # => [B, d_model]
        
        # (H) Classification layer
        logits = self.fc(x)  # => [B, num_classes]
        
        return logits
class Vanilla:
    def __init__(self, 
        loss_func,
        seq_len: int,
        num_classes: int,
        channels: int = 1,   # Number of input channels/features
        d_model: int = 4,
        nhead: int = 2,
        num_layers: int = 2,
        batch_first: bool = True, 
        to_cuda: bool = True,
    ):
        self.model = TimeSeriesTransformer(
            seq_len=seq_len,
            channels=channels,
            num_classes=num_classes,
            num_layers=num_layers,
            nhead=nhead, 
            d_model=d_model,
            batch_first=batch_first,
            to_cuda=to_cuda
        )
        self.to_cuda = to_cuda
        self.loss_func = loss_func
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
   
    def fit(
        self,
        X, Y, 
        X_val = None,
        Y_val = None,
        epochs=1, 
        batch_size=256, 
        shuffle=False, 
        drop_last=False, 
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

        
        
# ---------------------- Usage Example ----------------------
if __name__ == "__main__":
    # Hyperparameters
    SEQ_LEN = 50    # e.g., each time series has length=50
    CHANNELS = 1    # univariate
    D_MODEL = 64
    NHEAD = 8
    NUM_LAYERS = 2
    NUM_CLASSES = 2 # example number of classes
    BATCH_SIZE = 16

    # Instantiate the model
    model = TimeSeriesTransformer(seq_len=SEQ_LEN,
                                  d_model=D_MODEL,
                                  nhead=NHEAD,
                                  num_layers=NUM_LAYERS,
                                  num_classes=NUM_CLASSES,
                                  channels=CHANNELS,
                                  batch_first=True)
    
    # Create a random batch of shape [B, C, L]
    # e.g., univariate: shape = [16, 1, 50]
    x = torch.randn(BATCH_SIZE, CHANNELS, SEQ_LEN)
    
    # Forward pass
    logits = model(x)
    print("Logits shape:", logits.shape)  # [16, 3]

    # Define a loss, optimizer
    labels = torch.randint(high=NUM_CLASSES, size=(BATCH_SIZE,))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())
