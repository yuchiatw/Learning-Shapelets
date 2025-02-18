import torch
import torch.nn as nn
import torch.optim as optim

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
