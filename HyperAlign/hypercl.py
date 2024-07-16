import argparse
import random

import yaml
from tqdm import tqdm
import numpy as np
import torch

from models import HyperEncoder, hyperCL
from utils import drop_features, drop_incidence
from utils import load_data

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="CUDA initialization")


def train_cl(args, model, optimizer, num_nodes, num_edges, features, hyperedge_index, num_negs=None):


    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Hypergraph Augmentation
    hyperedge_index1 = drop_incidence(hyperedge_index, args.drop_incidence_rate)
    hyperedge_index2 = drop_incidence(hyperedge_index, args.drop_incidence_rate)
    x1 = drop_features(features, args.drop_feature_rate)
    x2 = drop_features(features, args.drop_feature_rate)

    # Encoder
    n1 = model(x1, hyperedge_index1, num_nodes, num_edges)
    n2 = model(x2, hyperedge_index2, num_nodes, num_edges)


    # Projection Head
    n1, n2 = model.node_projection(n1), model.node_projection(n2)

    loss = model.cl_loss(n1, n2, args.tau_n, num_negs=num_negs)
    

    loss.backward()
    optimizer.step()
    return loss.item()
