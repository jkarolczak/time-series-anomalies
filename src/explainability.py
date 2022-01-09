from itertools import combinations
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .model import Classifier, SineEstimator, infer


def anomaly_in_range(
    ts: np.ndarray,
    clf: Classifier
) -> List:
    clf_pattern = infer(ts, clf)
    ts_reconstructed = reconstruct(ts)
    result = [[] for _ in range(len(clf_pattern))]
    result_bisection = bisection(ts, ts_reconstructed, clf, clf_pattern)
    result_trisection = trisection(ts, ts_reconstructed, clf, clf_pattern)
    for i, (rb, rt) in enumerate(zip (result_bisection, result_trisection)):
        if not len(rb) and not len(rt):
            continue
        elif not len(rb):
            result[i] = rt
        elif not len(rt):
            result[i] = rb
        elif (rb[1] - rb[0]) < (rt[1] - rt[0]):
            result[i] = rb
        else:
            result[i] = rt   
    return result


def bisection(
    ts: np.ndarray,
    ts_reconstructed: np.ndarray,
    clf: Classifier,
    clf_pattern: np.ndarray
) -> List:
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


def trisection(
    ts: np.ndarray,
    ts_reconstructed: np.ndarray,
    clf: Classifier,
    clf_pattern: np.ndarray
) -> List:
    result = [[] for _ in range(len(clf_pattern))]
    for i in range(len(clf_pattern)):
        if clf_pattern[i]:
            left, right = 0, ts.shape[0]
            found = False
            while True:
                step = int((right - left) / 3)
                m1 = left + step
                m2 = left + 2 * step
                                
                lhs = ts.copy()
                lhs[left:m1, :] = ts_reconstructed[left:m1, :]
                lhs_infer = infer(lhs, clf)[i]
                if clf_pattern[i] and not lhs_infer:
                    right = m1
                    found = True                
                
                mhs = ts.copy()
                mhs[m1:m2, :] = ts_reconstructed[m1:m2, :]
                mhs_infer = infer(mhs, clf)[i]
                if clf_pattern[i] and not mhs_infer:
                    left = m1
                    right = m2
                    found = True
                
                rhs = ts.copy()
                rhs[m2:right, :] = ts_reconstructed[m2:right, :]
                rhs_infer = infer(rhs, clf)[i]
                if clf_pattern[i] and not rhs_infer:
                    left = m2
                    found = True
                    
                if found:
                    result[i] = [left, right]
                if all([lhs_infer, mhs_infer, rhs_infer]) or (right - left) < 6:                
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


def reconstruct(
    ts: np.ndarray, 
    silent: bool = True
) -> np.ndarray:
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


def plot_explanation(
    ts: np.ndarray,
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