import torch
import torch.nn as nn

class ShortFilter(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, 16, (1, 5), 1, 'same'),
            nn.ELU()
        )
        
    def forward(self, X):
        return self.layers(X)
    
class MultiFilter(nn.Module):
    def __init__(self, in_channels=36, pooling_size=2):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.AvgPool2d((pooling_size, 1)),
            nn.BatchNorm2d(in_channels)
        )
        self.cnns = nn.ModuleList([nn.Conv2d(in_channels, 24, (1, size), 1, 'same') for size in [32, 64, 96, 128, 192, 256]])
        self.bottleneck = nn.Conv2d(6*24, 36, 1, 1, 'same')
        
    def forward(self, X):
        X = self.preprocess(X)
        X = torch.cat([cnn(X) for cnn in self.cnns], axis=1)
        return self.bottleneck(X)

class TempoCNN(nn.Module):
    def __init__(self, output_size=256, window_size=256, bands=40):
        super().__init__()
        bottleneck_size = output_size//4
        self.layers = nn.Sequential(
            ShortFilter(1),
            ShortFilter(),
            ShortFilter(),
            MultiFilter(16, 5),
            MultiFilter(),
            MultiFilter(),
            MultiFilter(),
            nn.Flatten(),
            nn.BatchNorm1d(window_size*(bands//40)*36),
            nn.Dropout(0.5),
            nn.Linear(window_size*(bands//40)*36, bottleneck_size),
            nn.ELU(),
            nn.BatchNorm1d(bottleneck_size),
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.ELU(),
            nn.BatchNorm1d(bottleneck_size),
            nn.Linear(bottleneck_size, output_size),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, X):
        return self.layers(X)