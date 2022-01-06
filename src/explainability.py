from itertools import combinations
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .model import Classifier, infer, SineEstimator

def anomaly_in_range(
    ts: np.ndarray,
    clf: Classifier
) -> List:
    clf_pattern = infer(ts, clf)
    ts_reconstructed = reconstruct(ts)
    result = [[] for _ in range(len(clf_pattern))]
    for i in range(len(clf_pattern)):
        if clf_pattern[i]:
            left, right = 0, ts.shape[0]
            found = False
            while True:
                middle = int((left + right) / 2)
                
                lhs = ts.copy()
                lhs[left:middle, :] = ts_reconstructed[left:middle, :]
                lhs_infer = infer(lhs, clf)[i]
                if clf_pattern[i] and not lhs_infer:
                    right = middle
                    found = True                
                
                rhs = ts.copy()
                rhs[middle:right, :] = ts_reconstructed[middle:right, :]
                rhs_infer = infer(rhs, clf)[i]
                if clf_pattern[i] and not rhs_infer:
                    left = middle
                    found = True
                    
                if found and ((lhs_infer and rhs_infer) or (right - left) < 5):
                    result[i] = [left, right]
                    break
                if (right - left) < 5:
                    break
    return result


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


def plot_explanation(
    ts: np.ndarray,
    labels: np.ndarray,
    clf: Classifier
) -> None:
    on_ts = anomaly_on_ts(ts, clf)
    in_range = anomaly_in_range(ts, clf)
    plt.figure(figsize=(16, 9))
    plt.plot(ts)
    for a_ts, a_r in zip(on_ts, in_range):
        if len(a_ts) and len(a_r):
            plt.plot(range(a_r[0], a_r[1]), ts[a_r[0]:a_r[1], a_ts], color='red', linewidth=3)
    plt.show() 


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