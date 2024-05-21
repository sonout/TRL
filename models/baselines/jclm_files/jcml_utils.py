import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_sparse import SparseTensor




def jsd(z1, z2, pos_mask=None, calc_type="normal"):
    if calc_type == "optimized":
        # negative samples
        # NxE * (7*E)^T = Nx7
        # sample negative edge index
        row, col, _ = pos_mask.t().coo()
        pos_edge_index = torch.stack([row, col], dim=0)
        neg_edge_index = negative_sampling(
            pos_edge_index, z1.shape[0], num_neg_samples=z1.shape[0] * 7
        )
        sim_mat_pos = (z1[pos_edge_index[0]] * z2[pos_edge_index[1]]).sum(dim=1)
        sim_mat_neg = (z1[neg_edge_index[0]] * z2[neg_edge_index[1]]).sum(dim=1)
        E_pos = math.log(2.0) - F.softplus(-sim_mat_pos)
        E_neg = F.softplus(-sim_mat_neg) + sim_mat_neg - math.log(2.0)

        return (
            E_neg.sum() / neg_edge_index.shape[1]
            - (E_pos).sum() / pos_edge_index.shape[1]
        )
    else:
        neg_mask = 1 - pos_mask
        sim_mat = torch.mm(z1, z2.t())
        E_pos = math.log(2.0) - F.softplus(-sim_mat)
        E_neg = F.softplus(-sim_mat) + sim_mat - math.log(2.0)

        return (E_neg * neg_mask).sum() / neg_mask.sum() - (
            E_pos * pos_mask
        ).sum() / pos_mask.sum()


def nce(z1, z2, pos_mask):
    sim_mat = torch.mm(z1, z2.t())
    return nn.BCEWithLogitsLoss(reduction="none")(sim_mat, pos_mask).sum(1).mean()


def ntx(z1, z2, pos_mask, tau=0.5, normalize=False):
    if normalize:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    sim_mat = torch.mm(z1, z2.t())
    sim_mat = torch.exp(sim_mat / tau)
    return -torch.log(
        (sim_mat * pos_mask).sum(1) / sim_mat.sum(1) / pos_mask.sum(1)
    ).mean()


def node_node_loss(node_rep1, node_rep2, measure):
    num_nodes = node_rep1.shape[0]
    pos_mask = SparseTensor.eye(num_nodes, num_nodes)

    if measure == "jsd":
        return jsd(node_rep1, node_rep2, pos_mask, calc_type="optimized")
    elif measure == "nce":
        return nce(node_rep1, node_rep2, pos_mask)
    elif measure == "ntx":
        return ntx(node_rep1, node_rep2, pos_mask)


def seq_seq_loss(seq_rep1, seq_rep2, measure):
    batch_size = seq_rep1.shape[0]

    pos_mask = torch.eye(batch_size).cuda()

    if measure == "jsd":
        return jsd(seq_rep1, seq_rep2, pos_mask)
    elif measure == "nce":
        return nce(seq_rep1, seq_rep2, pos_mask)
    elif measure == "ntx":
        return ntx(seq_rep1, seq_rep2, pos_mask)


def node_seq_loss(node_rep, seq_rep, sequences, measure):
    batch_size = seq_rep.shape[0]
    num_nodes = node_rep.shape[0]

    pos_mask = torch.zeros((batch_size, num_nodes + 1)).cuda()
    for row_idx, row in enumerate(sequences):
        pos_mask[row_idx][row] = 1.0
    pos_mask = pos_mask[:, :-1]

    if measure == "jsd":
        return jsd(seq_rep, node_rep, pos_mask)
    elif measure == "nce":
        return nce(seq_rep, node_rep, pos_mask)
    elif measure == "ntx":
        return ntx(seq_rep, node_rep, pos_mask)


def weighted_ns_loss(node_rep, seq_rep, weights, measure):
    if measure == "jsd":
        return jsd(seq_rep, node_rep, weights)
    elif measure == "nce":
        return nce(seq_rep, node_rep, weights)
    elif measure == "ntx":
        return ntx(seq_rep, node_rep, weights)


def random_mask(x, mask_token, mask_prob=0.2):
    mask_pos = (
        torch.empty(x.size(), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < mask_prob
    )
    x = x.clone()
    x[mask_pos] = mask_token
    return