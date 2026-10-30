import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

from utils.utils import get_activation


class EGraphSage(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels,
                 activation, edge_mode, normalize_emb, aggr):
        super(EGraphSage, self).__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.edge_mode = edge_mode
        self.normalize_emb = normalize_emb

        if edge_mode == 1:
            self.message_lin = nn.Linear(in_channels + edge_channels, out_channels)
        else:
            self.message_lin = nn.Linear(in_channels + edge_channels, out_channels)

        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)
        self.message_activation = get_activation(activation)
        self.update_activation = get_activation(activation)

    def forward(self, x, edge_attr, edge_index):
        num_nodes = x.size(0)
        return self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(num_nodes, num_nodes))

    def message(self, x_j, edge_attr):
        m_j = torch.cat((x_j, edge_attr), dim=-1)
        return self.message_activation(self.message_lin(m_j))

    def update(self, aggr_out, x):
        aggr_out = self.update_activation(
            self.agg_lin(torch.cat((aggr_out, x), dim=-1)))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out
