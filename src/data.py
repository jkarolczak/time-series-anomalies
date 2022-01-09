from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        time_series: List[np.ndarray], 
        labels: List[np.ndarray]
    ):
        super().__init__()
        assert len(time_series) == len(labels)
        self.x = time_series
        self.y = labels
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx: int):
        x = torch.tensor(self.x[idx], dtype=torch.float32, requires_grad=True)
        y = torch.tensor(self.y[idx], dtype=torch.float32, requires_grad=True)
        return (x, y)
    
    
def dataset_split(
    dataset: Dataset,
    train_size: float = 0.75
) -> Tuple[Dataset, Dataset]:
    x, y = dataset.x, dataset.y
    x1, x2, y1, y2 = train_test_split(x, y, train_size=train_size)
    return Dataset(x1, y1), Dataset(x2, y2)
