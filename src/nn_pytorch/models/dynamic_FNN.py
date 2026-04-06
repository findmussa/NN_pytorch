import torch
import torch.nn as nn

ACTIVATIONS = {
    'relu':     nn.ReLU(),
    'tanh':     nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'elu'       : nn.ELU(),
    'gelu'      : nn.GELU(),
    'sigmoid'   : nn.Sigmoid()
}

class FNN(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, hidden_layers: list[int], activation: str = 'relu'):
        super().__init__()
        self.activation = ACTIVATIONS[activation]

        # build layers dynamically
        layer_sizes = [in_feat] + hidden_layers + [out_feat]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))   # hidden layer + activation
        x = self.layers[-1](x)              # out layer no activation
        return x
    

