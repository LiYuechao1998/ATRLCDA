import torch
import torch.nn.functional as F

from models.gnn_model import get_gnn
from models.prediction_model import build_decoder
from training.gae import GAE, sample_adj
from utils.utils import build_optimizer, get_known_mask, mask_edge


def half_pred(pred, n_edges_attr):
    return pred[: int(n_edges_attr / 2), 0]


class ATRLCDA(object):
    def __init__(self, data, args, device):
        self.args = args
        self.device = device
        self.num_ob = int(data.n_circ)
        self.model = get_gnn(data, args).to(device)
        self.decoder = build_decoder(args, args.node_dim * 2).to(device)
        self.gae_net = GAE(args.node_dim, 128, 32, F.relu).to(device)

        params = list(self.model.parameters()) + list(self.decoder.parameters())
        self.scheduler, self.opt = build_optimizer(args, params)
        self.gae_opt = torch.optim.Adam(
            self.gae_net.parameters(), lr=float(getattr(args, 'gae_lr', 1e-3)))

        pos_w = float(getattr(args, 'pos_weight', 1.0))
        if pos_w != 1.0:
            self.bce = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_w], device=device))
        else:
            self.bce = torch.nn.BCEWithLogitsLoss()

        self.x = data.x.clone().detach().to(device)
        self.struct_edge_index = data.struct_edge_index.clone().detach().to(device)
        self.struct_edge_attr = data.struct_edge_attr.clone().detach().to(device)
        self.train_edge_index = data.train_edge_index.clone().detach().to(device)
        self.train_labels = data.train_labels.clone().detach().to(device)
        self.obob_edge_index = data.obob_edge_index.to(device)
        self.obob_adj_norm = data.obob_adj_norm.to(device)
        self.obob_edge_weight = None
        self.known = float(getattr(args, 'known', 0.7))
        self.reconstruct_freq = int(getattr(args, 'reconstruct_freq', 100))

    def train_epoch(self, epoch):
        self.model.train()
        self.decoder.train()
        self.gae_net.train()
        self.opt.zero_grad()
        self.gae_opt.zero_grad()

        n_undirected = int(self.struct_edge_attr.shape[0] / 2)
        known_mask = get_known_mask(self.known, n_undirected).to(self.device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(
            self.struct_edge_index, self.struct_edge_attr, double_known_mask, True)
        if known_edge_index.shape[1] == 0:
            known_edge_index, known_edge_attr = (
                self.struct_edge_index, self.struct_edge_attr)

        x_embd = self.model(self.x, known_edge_attr, known_edge_index)

        if epoch % self.reconstruct_freq == 0:
            adj_logits = self.gae_net(
                self.obob_adj_norm, x_embd[:self.num_ob].clone())
            adj_new, _, edge_probs = sample_adj(adj_logits)
            nz = torch.nonzero(adj_new)
            if nz.numel() > 0:
                self.obob_edge_weight = edge_probs[adj_new.bool()]
                self.obob_edge_index = nz.T
            elif self.obob_edge_weight is None:
                self.obob_edge_weight = torch.ones(
                    self.obob_edge_index.shape[1], device=self.device)
        elif (epoch - 1) % self.reconstruct_freq == 0 and self.obob_edge_weight is not None:
            self.obob_edge_weight = self.obob_edge_weight.detach()

        if self.obob_edge_weight is None:
            self.obob_edge_weight = torch.ones(
                self.obob_edge_index.shape[1], device=self.device)
        x_sage = self.model.F_augmentation(
            x_embd[:self.num_ob], self.obob_edge_weight, self.obob_edge_index)
        x_embd = x_embd.clone()
        x_embd[:self.num_ob] = x_sage
        x_embd = self.model.node_post_mlp(x_embd)

        pred = self.decoder([
            x_embd[self.train_edge_index[0]],
            x_embd[self.train_edge_index[1]],
        ])
        scores = half_pred(pred, self.train_edge_index.shape[1])
        loss = self.bce(scores, self.train_labels)
        loss.backward()
        if epoch % self.reconstruct_freq == 0:
            self.gae_opt.step()
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, edge_index):
        self.model.eval()
        self.decoder.eval()
        x_embd = self.model(
            self.x, self.struct_edge_attr, self.struct_edge_index)
        if self.obob_edge_weight is None:
            self.obob_edge_weight = torch.ones(
                self.obob_edge_index.shape[1], device=self.device)
        x_sage = self.model.F_augmentation(
            x_embd[:self.num_ob], self.obob_edge_weight, self.obob_edge_index)
        x_embd = x_embd.clone()
        x_embd[:self.num_ob] = x_sage
        x_embd = self.model.node_post_mlp(x_embd)
        pred = self.decoder([x_embd[edge_index[0]], x_embd[edge_index[1]]])
        scores = half_pred(pred, edge_index.shape[1])
        return torch.sigmoid(scores)
