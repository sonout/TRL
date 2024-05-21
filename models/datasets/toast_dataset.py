import random
from operator import itemgetter

import networkx as nx
import numpy as np
import swifter
import torch
from _walker import random_walks as _random_walks
from scipy import sparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ToastDataset(Dataset):
    def __init__(self, data, edge_df, line_graph, config=None, train=True):
        self.edge_df = edge_df
        self.line_graph = line_graph
        self.train = train

        self.trajs = data["cpath"].values  # trajectory
        self.adj = nx.to_numpy_array(self.line_graph)
        np.fill_diagonal(self.adj, 0)
        self.traj_map = self._create_edge_emb_mapping()

        self.seq_len = 150
        self.mask_ratio = 0.20

        self.gen_new_walks(1)


    # tested index mapping is correct
    def _create_edge_emb_mapping(self):
        map = {}
        nodes = list(self.line_graph.nodes)
        for index, id in zip(self.edge_df.index, self.edge_df.fid):
            map[id] = nodes.index(index)
        # print(map == map2) # yields true

        return map

    def __len__(self):
        return self.trajs.shape[0]

    def __getitem__(self, item):
        if self.train:
            # case when training model on a dataset
            if random.random() > 0.5:
                walk = self.walks[item]
                if len(walk) > self.seq_len:
                    traj = self.cut_traj(walk)
                # shift by node ids by 2 to match pad and mask token
                walk = [w + 2 for w in walk]
                # mean_util = self.get_utilization(walk)
                is_traj = False
                (
                    traj_input,
                    traj_masked_tokens,
                    traj_masked_pos,
                    traj_masked_weights,
                    traj_label,
                ) = self.process_walk(walk)
            else:
                traj = list(
                    itemgetter(*self.trajs[item])(self.traj_map)
                )  # map to node ids
                if len(traj) > self.seq_len:
                    traj = self.cut_traj(traj)
                # shift by node ids by 2 to match pad and mask token
                traj = [t + 2 for t in traj]
                # mean_util = self.get_utilization(traj)
                is_traj = True
                (
                    traj_input,
                    traj_masked_tokens,
                    traj_masked_pos,
                    traj_masked_weights,
                    traj_label,
                ) = self.random_word(traj)

            traj_input = traj_input
            traj_label = traj_label
            input_mask = [1] * len(traj_input)
            length = [len(traj_input)]

            masked_lenth = len(traj_masked_tokens)
            padding = [0 for _ in range(self.seq_len - len(traj_input))]
            traj_input.extend(padding)
            input_mask.extend(padding)
            traj_label.extend(padding)

            max_pred = int(self.seq_len * self.mask_ratio)
            if max_pred > masked_lenth:
                padding = [0] * (max_pred - masked_lenth)
                traj_masked_tokens.extend(padding)
                traj_masked_pos.extend(padding)
                traj_masked_weights.extend(padding)
            else:
                traj_masked_tokens = traj_masked_tokens[:max_pred]
                traj_masked_pos = traj_masked_pos[:max_pred]
                traj_masked_weights = traj_masked_weights[:max_pred]

            return (
                traj_input,
                input_mask,
                traj_masked_pos,
                length,
                is_traj,
                traj_masked_tokens,
                traj_masked_weights,
            )

        else:
            # case when evaluating model on a dataset
            traj = list(itemgetter(*self.trajs[item])(self.traj_map))  # map to node ids
            if len(traj) > self.seq_len:
                traj = self.cut_traj(traj)
            # shift by node ids by 2 to match pad and mask token
            traj = [t + 2 for t in traj]
            traj_input = traj
            input_mask = [1] * len(traj_input)
            length = [len(traj_input)]

            padding = [0 for _ in range(self.seq_len - len(traj_input))]
            traj_input.extend(padding)
            input_mask.extend(padding)

            return (
                traj_input,
                input_mask,
                [],
                length,
                [],
                [],
                [],
            )

    def random_word(self, sentence):
        tokens = sentence
        output_label = []

        mask_len = int(len(tokens) * self.mask_ratio)
        start_loc = round(len(tokens) * (1 - self.mask_ratio))
        # round(len(tokens) * random.random() * (1 - self.mask_ratio))

        masked_pos = list(range(start_loc, start_loc + mask_len))
        masked_tokens = tokens[start_loc : start_loc + mask_len]
        masked_weights = [1] * len(masked_tokens)

        for i, token in enumerate(tokens):
            if i >= start_loc and i < start_loc + mask_len:
                tokens[i] = 1
                output_label.append(token)
            else:
                output_label.append(0)

        assert len(tokens) == len(output_label)

        return tokens, masked_tokens, masked_pos, masked_weights, output_label

    def process_walk(self, walk):
        tokens = walk
        output_label = []

        mask_len = int(len(tokens) * self.mask_ratio)
        start_loc = round(len(tokens) * (1 - self.mask_ratio))
        # round(len(tokens) * random.random() * (1 - self.mask_ratio))

        masked_pos = list(range(start_loc, start_loc + mask_len))
        masked_tokens = tokens[start_loc : start_loc + mask_len]
        masked_weights = [0] * len(masked_tokens)

        for i, token in enumerate(tokens):
            if i >= start_loc and i < start_loc + mask_len:
                tokens[i] = 1
                output_label.append(token)
            else:
                output_label.append(0)

        assert len(tokens) == len(output_label)

        return tokens, masked_tokens, masked_pos, masked_weights, output_label

    def cut_traj(self, traj):
        start_idx = int((len(traj) - self.seq_len) * random.random())
        return traj[start_idx : start_idx + self.seq_len]

    def gen_new_walks(self, num_walks):
        self.walks = ToastDataset.traj_walk(
            self.adj,
            start=np.arange(len(self.line_graph.nodes)),
            walks_per_node=num_walks,
        )
        random.shuffle(self.walks)

    @staticmethod
    def traj_walk(adj, start, walks_per_node):
        A = sparse.csr_matrix(adj)
        indptr = A.indptr.astype(np.uint32)
        indices = A.indices.astype(np.uint32)
        data = A.data.astype(np.float32)
        walk_length = random.randint(10, 100)
        walks = (
            _random_walks(indptr, indices, data, start, walks_per_node, walk_length + 1)
            .astype(int)
            .tolist()
        )

        return walks

    @staticmethod
    def collate_custom(batch):
        (
            traj_input,
            input_mask,
            masked_pos,
            length,
            is_traj,
            masked_tokens,
            masked_weights,
        ) = zip(*batch)

        return (
            torch.tensor(traj_input),
            torch.tensor(input_mask),
            torch.tensor(masked_pos),
            torch.tensor(length),
            torch.tensor(is_traj),
            torch.tensor(masked_tokens),
            torch.tensor(masked_weights),
        )
