from enum import Enum
from operator import itemgetter
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from .baselines import *
from .token_embs.src.cell.cellspace import CellSpace

MODELS_PACKAGE_NAME = "models"
DATASETS_PACKAGE_NAME = "models"


def meters2lonlat(x, y):
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = math.exp(y / 3189068.5)
    lat = math.asin((t - 1) / (t + 1)) / 0.017453292519943295
    return lon, lat

# ref: TrjSR
def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def merc2cell2(src, cs: CellSpace):
    # convert and remove consecutive duplicates
    tgt = [ (cs.get_cellid_by_point(*p), p) for p in src]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i-1][0]]
    tgt, tgt_p = zip(*tgt)
    return tgt, tgt_p

# tested index mapping is correct
def create_edge_emb_mapping(edge_df, LG):
    map = {}
    nodes = list(LG.nodes)
    for index, id in zip(edge_df.index, edge_df.fid):
        map[id] = nodes.index(index)
    # print(map == map2) # yields true
    return map

def cut_traj(traj, seq_len):
        start_idx = int((len(traj) - seq_len) * random.random())
        return traj[start_idx : start_idx + seq_len]

def map_trajectory_to_road_embeddings(emb, batch, mask, map):
    """
    Transform (batch_size, seq_length, 1) to (batch_size, seq_length, emb_size)
    """
    # res = torch.zeros((batch.shape[0], batch.shape[1], emb.shape[-1]))
    # for i, seq in enumerate(batch):
    #     emb_ids = itemgetter(*seq[mask[i]].tolist())(map)
    #     res[i, mask[i], :] = torch.Tensor(emb[emb_ids, :])

    # batch = batch.detach().cpu().apply_(map.__getitem__)
    # select_t = emb.weight.unsqueeze(0).repeat(batch.shape[0], 1, 1)
    # index_t = batch.unsqueeze(-1).expand(batch.shape[0], batch.shape[1], 128)
    # res = torch.gather(select_t, dim=1, index=index_t)
    # res[~mask.unsqueeze(-1).expand(mask.shape[0], mask.shape[1], 128)] = 0

    res = emb(batch)

    # print(res2.shape, res.shape, torch.equal(res, res2), mask.shape)

    return res


def contrastive_loss_simclr(z1, z2, temperature: float = 0.05):
    """
    Args:
        z1(torch.tensor): (batch_size, d_model)
        z2(torch.tensor): (batch_size, d_model)
    Returns:
    """
    assert z1.shape == z2.shape
    batch_size, d_model = z1.shape
    features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(
        labels.shape[0], -1
    )  # [batch_size * 2, 1]

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(
        similarity_matrix.shape[0], -1
    )  # [batch_size * 2, 2N-2]

    logits = torch.cat(
        [positives, negatives], dim=1
    )  # (batch_size * 2, batch_size * 2 - 1)
    labels = torch.zeros(
        logits.shape[0], dtype=torch.long
    ).cuda()  # (batch_size * 2, 1)
    logits = logits / temperature

    return logits, labels


