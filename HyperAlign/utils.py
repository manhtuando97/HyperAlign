import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from hyperFeat.utils import *

import random
from itertools import permutations
from typing import Optional, Callable

import numpy as np

from torch.nn import Parameter

from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_scatter import scatter_add

# prediction of alignment
def prediction(args, n_embd1, n_embd2, node_list1, node_list2):
    norm_1 = torch.nn.functional.normalize(n_embd1, dim=1, p=2)
    norm_2 = torch.nn.functional.normalize(n_embd2, dim=1, p=2)

    cossim = torch.mm(norm_2, norm_1.transpose(0,1))

    _, top_idx1 = torch.topk(cossim, 1, dim=1)

    h = open('{}/{}-{}.txt'.format(args.pred_output, args.dataset1, args.dataset2), 'w')
    for n1 in node_list1:
        h.write('{} {}\n'.format(str(n1), str((top_idx1[n1].item()))))
    h.close()

# for loading data
def load_data(hypergraph_input_dir, feature_input_dir, name):
    f = open('{}/{}.emb'.format(feature_input_dir, name), 'r')
    h = open('{}/{}.txt'.format(hypergraph_input_dir, name), 'r')

    # load features
    f.readline()

    feature_dict = {}
    for line in f:
        lines = line.split(' ')
        feature_dict[int(lines[0])] = np.array(lines[1:], dtype = float)

    node_list = list(feature_dict.keys())
    max_node_id = max(feature_dict)
    for missing_id in range(max_node_id):
        if missing_id not in feature_dict.keys():
            feature_dict[missing_id] = np.zeros(len(lines) - 1)

    feat = [feature_dict[id] for id in sorted(feature_dict.keys())]
    features = torch.tensor(np.array(feat), dtype=torch.float32)
    f.close()


    node_idx = []
    hyperedge_idx = []

    hyperedge_count = 0
    for line in h:
        lines = line.split(' ')
        for node in lines:
            node_idx.append(int(node))
            hyperedge_idx.append(hyperedge_count)

        hyperedge_count += 1

    hyperedge_index = torch.tensor([node_idx, hyperedge_idx])

    dictionary = dict()
    for i in node_idx:
        dictionary[i] = i

    return features.shape[0], hyperedge_count, features, hyperedge_index, node_list, dictionary

# for HyperCL
def drop_features(x: Tensor, p: float):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]


def drop_incidence(hyperedge_index: Tensor, p: float = 0.2):
    if p == 0.0:
        return hyperedge_index

    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p

    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)
    return hyperedge_index



# feature reconstruction loss
def feature_reconstruct_loss(embd, x, recon_model):
        recon_x = recon_model(embd)
        return torch.norm(recon_x - x, dim=1, p=2).mean()

# predict alignment based on node having the most similar embedding
def prediction(args, n_embd1, n_embd2, node_list1, node_list2, k = 1):
    norm_1 = torch.nn.functional.normalize(n_embd1, dim=1, p=2)
    norm_2 = torch.nn.functional.normalize(n_embd2, dim=1, p=2)
    #norm_1 = n_embd1
    #norm_2 = n_embd2
    cossim = torch.mm(norm_2, norm_1.transpose(0,1))
    #print(cossim)
    #n_nodes = n_embd1.shape[0]
    #print(n_nodes)
    n_embd1, n_embd2 = satinary_check(args, n_embd1, n_embd2)
    _, top_idx2 = torch.topk(cossim, 1, dim=1)
    h = open('{}/{}-{}.txt'.format(args.pred_output, args.dataset1, args.dataset2), 'w')
    for n2 in node_list2:
        _, top_idx2 = torch.topk(cossim, n_embd2[1], dim=1)

        #for candidate in top_idx2[n2].item().tolist():
            #cossim[:, top_idx2[n2].item()] = 0
            #cossim[:, top_idx2[n2].item()] = 0
        #node_string = [str(node) for node in top_idx2[n2].tolist()]
        #answer_string = ' '.join(node_string)

        if n2 in top_idx2[n2].tolist():
            answer_string = '{} {}\n'.format(str(n2), str(n2))
            cossim[:, n2] = 0

        else:
            _, top_idx2 = torch.topk(cossim, 1, dim=1)
            answer_string = '{} {}\n'.format(str((top_idx2[n2].item())), str(n2))
            cossim[:, top_idx2[n2].item()] = -1

        #cossim[:, top_idx2[n2].item()] = 0
        # comment
        #h.write('{} {}\n'.format(str((top_idx2[n2].item())), str(n2)))
        h.write('{}'.format(answer_string))
    h.close()
