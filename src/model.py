import os
from datetime import datetime

import torch
import torch.nn as nn


class SequentialLSTM(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        batch_first: bool = True,
        bidirectional: bool = True
    ):
        super(SequentialLSTM, self).__init__()
        if bidirectional:
            assert not out_channels % 2
            out_channels = int(out_channels / 2)
        self.lstm = nn.LSTM(
            input_size=in_channels, 
            hidden_size=out_channels, 
            batch_first=batch_first, 
            bidirectional=bidirectional
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        return x


class Classifier(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 5
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            SequentialLSTM(in_channels, 16),
            SequentialLSTM(16, 32),
            SequentialLSTM(32, 32),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * 32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, out_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor):
        x = self.encoder(x)      
        x_min = x.min(axis=1).values
        x_mean = x.mean(axis=1)
        x_max = x.max(axis=1).values
        x = torch.cat(
            [x_min, x_mean, x_max],
            dim=1
        )
        x = self.classifier(x)
        return x


class SineEstimator(nn.Module):
    def __init__(
        self,
        ts: torch.Tensor
    ):
        super().__init__()
        self.len = ts.shape[0]
        self.vars = ts.shape[1]
        self.a = nn.Parameter(
            torch.tensor([[0.2], [0.3], [0.1]], dtype=torch.float32)
        )
        self.b = nn.Parameter(
            torch.ones((3, 1), dtype=torch.float32)
        )
        self.x = torch.tensor([
            [list(range(self.len))] * self.vars
        ], dtype=torch.float32)
        
    def forward(self) -> torch.Tensor:
        result = torch.sin(self.a * self.x + self.b)
        return result.swapaxes(-1, -2).squeeze(0)


def serialize(
    model: nn.Module,
    epoch: int,
    directory: str = "models" 
) -> None:
    os.makedirs(directory, exist_ok=True)
    time = str(datetime.now()).replace(' ', '-')
    file_name = f'{time}-epoch-{epoch}.pt'
    file_path = os.path.join(directory, file_name)
    torch.save(model.state_dict(), file_path)
    

def deserialize(
    file_name: str,
    directory: str = "models" ,
    model: nn.Module = Classifier
) -> nn.Module:
    file_path = os.path.join(directory, file_name)
    state_dict = torch.load(file_path)
    model = model()
    model.load_state_dict(state_dict)
    return model