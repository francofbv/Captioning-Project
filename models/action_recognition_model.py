import torch
import torch.nn as nn
from cnn import CNN
from transformer import Transformer

class ActionRecognitionModel(nn.Module):
    def __init__(self, 
                 num_classes=101,
                 d_model=2048,
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=8192,
                 dropout=0.1):
        """
        Action Recognition Model combining CNN and Transformer
        Args:
            num_classes (int): Number of action classes
            d_model (int): Dimension of frame embeddings
            nhead (int): Number of attention heads
            num_encoder_layers (int): Number of transformer encoder layers
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout rate
        """
        super(ActionRecognitionModel, self).__init__()
        
        # CNN for frame feature extraction
        self.cnn = CNN()
        
        # Transformer for temporal modeling
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=num_classes
        )
        
    def forward(self, x, mask=None):
        """
        Forward pass
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, channels, height, width]
            mask (torch.Tensor, optional): Mask for transformer of shape [batch_size, seq_len]
        Returns:
            torch.Tensor: Action class logits of shape [batch_size, num_classes]
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # Reshape input for CNN
        # Combine batch and sequence dimensions
        x = x.view(-1, channels, height, width)
        
        # Extract frame embeddings using CNN
        # Shape: [batch_size * seq_len, d_model]
        frame_embeddings = self.cnn(x)
        
        # Reshape back to include sequence dimension
        # Shape: [batch_size, seq_len, d_model]
        frame_embeddings = frame_embeddings.view(batch_size, seq_len, -1)
        
        # Pass through transformer for temporal modeling
        # Shape: [batch_size, num_classes]
        action_logits = self.transformer(frame_embeddings, mask)
        
        return action_logits
    
    def predict(self, x, mask=None):
        """
        Predict action class
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, channels, height, width]
            mask (torch.Tensor, optional): Mask for transformer of shape [batch_size, seq_len]
        Returns:
            torch.Tensor: Action class probabilities of shape [batch_size, num_classes]
        """
        # Get logits
        logits = self.forward(x, mask)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        
        return probs
    
    def get_attention_weights(self, x, mask=None):
        """
        Get attention weights from transformer
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, channels, height, width]
            mask (torch.Tensor, optional): Mask for transformer of shape [batch_size, seq_len]
        Returns:
            torch.Tensor: Attention weights of shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # Reshape input for CNN
        x = x.view(-1, channels, height, width)
        
        # Extract frame embeddings
        frame_embeddings = self.cnn(x)
        frame_embeddings = frame_embeddings.view(batch_size, seq_len, -1)
        
        # Get attention weights from transformer
        attention_weights = self.transformer.get_attention_weights(frame_embeddings, mask)
        
        return attention_weights
