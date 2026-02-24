import torch
import pandas as pd
import numpy as np


class TrackXYLoader(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        df = pd.read_csv(path)
        self.data = df.to_numpy()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
