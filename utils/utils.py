import torch
import torch.optim as optim


def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    if activation == 'prelu':
        return torch.nn.PReLU()
    if activation == 'tanh':
        return torch.nn.Tanh()
    if activation is None or activation == 'none':
        return torch.nn.Identity()
    raise NotImplementedError


def build_optimizer(args, params):
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(
            filter_fn, lr=args.lr, momentum=0.95, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    if args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return scheduler, optimizer


def get_known_mask(known_prob, edge_num):
    return (torch.FloatTensor(edge_num, 1).uniform_() < known_prob).view(-1)


def mask_edge(edge_index, edge_attr, mask, remove_edge):
    edge_index = edge_index.clone().detach()
    edge_attr = edge_attr.clone().detach()
    if remove_edge:
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]
    else:
        edge_attr[~mask] = 0.
    return edge_index, edge_attr
