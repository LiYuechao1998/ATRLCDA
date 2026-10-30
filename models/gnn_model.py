import torch
import torch.nn as nn

from models.egsage import EGraphSage
from models.psage import PGraphSage
from utils.utils import get_activation


def get_gnn(data, args):
    model_types = args.model_types.split('_')
    if args.norm_embs is None:
        norm_embs = [True] * len(model_types)
    else:
        norm_embs = list(map(bool, map(int, args.norm_embs.split('_'))))
    if args.post_hiddens is None:
        post_hiddens = [args.node_dim]
    else:
        post_hiddens = list(map(int, args.post_hiddens.split('_')))
    model = GNNStack(
        data.num_node_features, data.edge_attr_dim,
        args.node_dim, args.edge_dim, args.edge_mode,
        model_types, args.dropout, args.gnn_activation,
        args.concat_states, post_hiddens, norm_embs, args.aggr)
    return model


class GNNStack(torch.nn.Module):
    def __init__(self, node_input_dim, edge_input_dim,
                 node_dim, edge_dim, edge_mode,
                 model_types, dropout, activation,
                 concat_states, node_post_mlp_hiddens,
                 normalize_embs, aggr):
        super(GNNStack, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.concat_states = concat_states
        self.model_types = model_types
        self.gnn_layer_num = len(model_types)
        self.convs = self.build_convs(
            node_input_dim, edge_input_dim, node_dim, edge_dim, edge_mode,
            model_types, normalize_embs, activation, aggr)
        self.node_post_mlp = self.build_node_post_mlp(
            node_dim, node_dim, node_post_mlp_hiddens, dropout, activation)
        self.edge_update_mlps = self.build_edge_update_mlps(
            node_dim, edge_input_dim, edge_dim, self.gnn_layer_num, activation)
        self.sage = self.build_friend_network_convs(
            node_dim, node_dim, activation, normalize_embs, aggr)

    def build_friend_network_convs(self, input_dim, output_dim, activation,
                                   normalize_embs, aggr):
        convs = nn.ModuleList()
        convs.append(PGraphSage(
            input_dim, output_dim, activation, normalize_embs[0], aggr))
        return convs

    def build_node_post_mlp(self, input_dim, output_dim, hidden_dims,
                            dropout, activation):
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                get_activation(activation),
                nn.Dropout(dropout),
            ))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def build_convs(self, node_input_dim, edge_input_dim, node_dim, edge_dim,
                    edge_mode, model_types, normalize_embs, activation, aggr):
        convs = nn.ModuleList()
        convs.append(EGraphSage(
            node_input_dim, node_dim, edge_input_dim, activation,
            edge_mode, normalize_embs[0], aggr))
        for l in range(1, len(model_types)):
            convs.append(EGraphSage(
                node_dim, node_dim, edge_dim, activation,
                edge_mode, normalize_embs[l], aggr))
        return convs

    def build_edge_update_mlps(self, node_dim, edge_input_dim, edge_dim,
                               gnn_layer_num, activation):
        edge_update_mlps = nn.ModuleList()
        edge_update_mlps.append(nn.Sequential(
            nn.Linear(node_dim + node_dim + edge_input_dim, edge_dim),
            get_activation(activation),
        ))
        for _ in range(1, gnn_layer_num):
            edge_update_mlps.append(nn.Sequential(
                nn.Linear(node_dim + node_dim + edge_dim, edge_dim),
                get_activation(activation),
            ))
        return edge_update_mlps

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0], :]
        x_j = x[edge_index[1], :]
        return mlp(torch.cat((x_i, x_j, edge_attr), dim=-1))

    def forward(self, x, edge_attr, edge_index):
        for l, conv in enumerate(self.convs):
            x = conv(x, edge_attr, edge_index)
            edge_attr = self.update_edge_attr(
                x, edge_attr, edge_index, self.edge_update_mlps[l])
        return x

    def F_augmentation(self, x, edge_attr, edge_index):
        for conv in self.sage:
            x = conv(x, edge_attr, edge_index)
        return x
