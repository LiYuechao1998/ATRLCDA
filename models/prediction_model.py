import numpy as np
import torch
import torch.nn as nn

from utils.utils import get_activation


class MLPNet(torch.nn.Module):
    def __init__(self, input_dims, output_dim, hidden_layer_sizes=(64,),
                 hidden_activation='relu', output_activation=None, dropout=0.):
        super(MLPNet, self).__init__()
        layers = nn.ModuleList()
        input_dim = np.sum(input_dims)
        for layer_size in hidden_layer_sizes:
            layers.append(nn.Sequential(
                nn.Linear(input_dim, layer_size),
                get_activation(hidden_activation),
                nn.Dropout(dropout),
            ))
            input_dim = layer_size
        layers.append(nn.Sequential(
            nn.Linear(input_dim, output_dim),
            get_activation(output_activation),
        ))
        self.layers = layers

    def forward(self, inputs):
        if torch.is_tensor(inputs):
            inputs = [inputs]
        h = torch.cat(inputs, -1)
        for layer in self.layers:
            h = layer(h)
        return h


def build_decoder(args, input_dim):
    if args.impute_hiddens == '':
        hidden = []
    else:
        hidden = list(map(int, args.impute_hiddens.split('_')))
    return MLPNet(
        input_dim, 1,
        hidden_layer_sizes=hidden,
        hidden_activation=args.impute_activation,
        dropout=args.dropout)
