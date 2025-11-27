import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model,feature_dim):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        


        self.projection = nn.Linear(feature_dim,d_model)
    
    def forward(self,x):
        return self.projection(x)*math.sqrt(self.d_model)


# model_components.py (continued)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # Create a vector of positions (0, 1, ... seq_len-1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        
        # Calculate the division term (log space for numerical stability as seen in video)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply Sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply Cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as a buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # Add the positional encoding to the embeddings
        # We slice pe to [:x.shape[1]] in case the input is shorter than max seq_len
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)



class LayerNormalization(nn.Module):
    def __init__(self,eps:float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean  = x.mean(dim = -1,keepdim = True)
        std = x .std(dim = -1,keepdim = True)

        return self.alpha *(x -mean)/std+self.bias



class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # First linear layer expands the dimension
        self.linear_1 = nn.Linear(d_model, d_ff) 
        self.dropout = nn.Dropout(dropout)
        # Second linear layer compresses it back
        self.linear_2 = nn.Linear(d_ff, d_model) 

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        # We use ReLU as the activation function
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# model_components.py (continued)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        
        # We ensure d_model can be evenly divided by heads
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo (Output projection)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # (Batch, h, Seq_Len, d_k) @ (Batch, h, d_k, Seq_Len) --> (Batch, h, Seq_Len, Seq_Len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # If mask is provided (very important for Decoder to not see future prices), apply it
        if mask is not None:
            # Replace masked values with a very small number (-infinity) so softmax makes them 0
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_scores = attention_scores.softmax(dim=-1) # (Batch, h, Seq_Len, Seq_Len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        # (Batch, h, Seq_Len, Seq_Len) @ (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, Seq_Len, d_model)
        key = self.w_k(k)   # (Batch, Seq_Len, d_model)
        value = self.w_v(v) # (Batch, Seq_Len, d_model)

        # Split into heads
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Concatenate heads
        # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # Final output projection
        return self.w_o(x)

# model_components.py (continued)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features) # Using our custom LayerNormalization class

    def forward(self, x, sublayer):
        # We normalize x, pass it to the sublayer (Attention or FeedForward), 
        # apply dropout, and add it back to the original x.
        return x + self.dropout(sublayer(self.norm(x)))