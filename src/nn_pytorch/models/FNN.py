import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, in_feat: int, h1: int = 64, h2: int = 32, out_feat: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(in_feat, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

