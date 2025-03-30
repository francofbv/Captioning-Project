import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    '''
    Positional Encoding for the Transformer model
    Input: 2048-dimensional frame embedding
    Output: 2048-dimensional frame embedding with positional encoding
    '''
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    '''
    Model architecture: Transformer for caption generation
    Input: 2048-dimensional frame embedding
    Output: 101-dimensional action class probability distribution
    '''
    def __init__(self, 
                 d_model=2048,          # Dimension of frame embeddings
                 nhead=8,               # Number of attention heads
                 num_encoder_layers=6,  # Number of transformer encoder layers
                 dim_feedforward=8192,  # Dimension of feedforward network
                 dropout=0.1,           # Dropout rate
                 num_classes=101):      # Number of action classes
        super(Transformer, self).__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global average pooling over sequence length
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x