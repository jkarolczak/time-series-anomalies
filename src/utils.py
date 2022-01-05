from itertools import combinations
from typing import List

import numpy as np
import torch
import torch.nn as nn

from .model import Classifier, SineEstimator


def reconstruct(ts: np.ndarray, silent: bool = True) -> np.ndarray:
    y = torch.tensor(ts, dtype=torch.float32, requires_grad=True)

    sine = SineEstimator(y)
    epochs = 10000
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(sine.parameters(), lr=1e-3)

    ls = []
    sine.train()
    for e in range(epochs):
        optimizer.zero_grad()
        y_pred = sine()
        loss = criterion(y, y_pred)
        loss.backward()
        optimizer.step()
        ls.append(loss.detach().numpy())
        
        if e >= 10 and all(np.array(ls[-11:-1]) <= ls[-1]):
            if not silent:
                print(f"Epoch {e} - early stopping")
            break
        
    sine.eval()
    return sine().detach().numpy()


def infer(
    ts: torch.Tensor,
    clf: Classifier,
    device: torch.device = None
) -> np.ndarray:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(ts, dtype=torch.float32, device=device).unsqueeze(0)
    return (clf(x).squeeze(0).detach().cpu().numpy() > 0.5)


def anomaly_on_ts(
    ts: np.ndarray,
    clf: Classifier
) -> List:
    clf_pattern = infer(ts, clf)
    result = [[] for _ in range(len(clf_pattern))]
    ts_reconstructed = reconstruct(ts)
    
    for idx in list(range(ts.shape[1])) + list(combinations(range(ts.shape[1]), 2)):
        ts_temp = ts.copy()
        ts_temp[:, idx] = ts_reconstructed[:, idx]
        clf_temp = infer(ts_temp, clf)
        for j in range(len(clf_pattern)):
            try: len(idx)
            except: idx = [idx]
            if clf_pattern[j] and not clf_temp[j] and not any([idx in result[j] for idx in list(idx)]):
                result[j].extend(idx)
        
    return result