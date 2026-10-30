import torch
import torch.nn as nn

try:
    import pyro
except ImportError:
    pyro = None


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation, bias=False):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        self.b = nn.Parameter(torch.FloatTensor(output_dim)) if bias else None
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        x = adj @ (h @ self.W)
        if self.b is not None:
            x = x + self.b
        if self.activation is not None:
            x = self.activation(x)
        return x


class GAE(nn.Module):
    def __init__(self, dim_feats, dim_h, dim_z, activation):
        super(GAE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(dim_feats, dim_h, None, bias=False))
        self.layers.append(GCNLayer(dim_h, dim_z, activation, bias=False))

    def forward(self, adj, features):
        hidden = self.layers[0](adj, features)
        z = self.layers[1](adj, hidden)
        return z @ z.T


def sample_adj(adj_logits):
    adj_logits = torch.relu(adj_logits)
    adj_logits_ = adj_logits / torch.max(adj_logits)
    if pyro is not None:
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(
            temperature=0.2, probs=adj_logits_).rsample()
    else:
        hard = (torch.rand_like(adj_logits_) < adj_logits_).float()
        adj_sampled = adj_logits_ + (hard - adj_logits_).detach()
    adj_sampled = adj_sampled.triu(1)
    return adj_sampled, adj_sampled + adj_sampled.T, adj_logits_
