
import torch
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import numpy as np
import pandas as pd

class FinancialDataset(Dataset):
    def __init__(self, ticker, seq_len):
        """
        Args:
            ticker (str): The stock symbol (e.g., 'AAPL').
            seq_len (int): How many past days the model looks at.
        """
        self.seq_len = seq_len
        

        print(f"Downloading daa {ticker}...")
        data = yf.download(ticker, start="2010-01-01", end="2023-01-01")

        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Log returns for stability or Standardization
        # Here we simple standardize: (X - mean) / std
        self.mean = data.mean()
        self.std = data.std()
        self.data_normalized = (data - self.mean) / self.std
        
        # Convert to float32 tensor
        self.data_tensor = torch.tensor(self.data_normalized.values, dtype=torch.float32)

    def __len__(self):
        # We need enough data for the input sequence + target sequence (1 step ahead prediction for simplicity here)
        return len(self.data_tensor) - self.seq_len - 1

    def __getitem__(self, idx):
        # Input: The sequence of 'seq_len' days
        enc_input = self.data_tensor[idx : idx + self.seq_len]
        
        # Target: The sequence shifted by 1 (classic Transformer training)
        # In a real forecast, we might predict just the next 1 day, but for this 
        # structure, we will treat it like translation: 
        # Source (Past 60 days) -> Target (Next 60 days shifted)
        dec_input = self.data_tensor[idx + 1 : idx + self.seq_len + 1]
        
        # Create masks (We will implement the mask functions in part 2, 
        # but we place holders here similar to the video)
        # For finance, we assume fully visible encoder, causal decoder.
        
        return {
            "encoder_input": enc_input, # (seq_len, feature_dim)
            "decoder_input": dec_input, # (seq_len, feature_dim)
            # We will handle labels (the actual future price) in the training loop
            "label": dec_input 
        }

def get_ds(config):
    ds = FinancialDataset(config['datasource'], config['seq_len'])
    # Split 90/10
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    return train_dataloader, val_dataloader