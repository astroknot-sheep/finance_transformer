from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 60,  # Lookback window (e.g., past 60 days)
        "d_model": 512, # The dimension of the vector inside the transformer
        "d_ff": 2048,   # Feed forward hidden dimension
        "n_heads": 8,   # Number of attention heads
        "n_layers": 6,  # Number of encoder/decoder layers
        "dropout": 0.1,
        "feature_dim": 5, # Open, High, Low, Close, Volume
        "datasource": "AAPL", # Ticker symbol
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "experiment_name": "runs/tmodel"
    }
def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)