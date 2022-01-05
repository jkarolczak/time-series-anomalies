import warnings
from typing import Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor
) -> Tuple[float, float, float, float, float]:
    
    loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
    y_true, y_pred = y_true.int().numpy(), (y_pred > 0.8).int().numpy()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc = sum([accuracy_score(y1, y2) for y1, y2 in zip(y_true, y_pred)]) / len(y_true)
        prec = sum([precision_score(y1, y2) for y1, y2 in zip(y_true, y_pred)]) / len(y_true)
        rec = sum([recall_score(y1, y2) for y1, y2 in zip(y_true, y_pred)]) / len(y_true)
        f1 = sum([f1_score(y1, y2) for y1, y2 in zip(y_true, y_pred)]) / len(y_true)
    
    return (loss, acc, prec, rec, f1)
