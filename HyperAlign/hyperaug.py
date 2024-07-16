import scipy

import numpy as np
import torch
import itertools
seed = 1
torch.manual_seed(seed)
import torch.nn.functional as F
import torch_geometric
import time
import argparse
import pickle

from utils import feature_reconstruct_loss, load_data
from gan import Discriminator, transformation, ReconDNN, notrans


# train adversarial
def train_adv(model, trans, optimizer_trans, dataset1, dataset2, epochs, alpha, lr_wd, lr_recon, augmentation_t=0):
    torch.seed()

    num_nodes1, num_edges1, features1, hyperedge_index1, Dn1 = dataset1
    num_nodes2, num_edges2, features2, hyperedge_index2, Dn2 = dataset2

    feature_size = features1.shape[1]
    feature_output_size = features1.shape[1]

    discriminator = Discriminator(model.encoder.node_dim, model.encoder.node_dim)
    optimizer_wd = torch.optim.Adam(discriminator.parameters(), lr=lr_wd, weight_decay=5e-4)

    recon_model2 = ReconDNN(model.encoder.node_dim, feature_size)
    recon_model1 = ReconDNN(model.encoder.node_dim, feature_size)
    optimizer_recon2 = torch.optim.Adam(recon_model2.parameters(), lr=lr_recon, weight_decay=5e-4)
    optimizer_recon1 = torch.optim.Adam(recon_model1.parameters(), lr=lr_recon, weight_decay=5e-4)

    batch_size_align = 128

    updated_dataset1 = num_nodes1, num_edges1, features1, hyperedge_index1
    updated_dataset2 = num_nodes2, num_edges2, features2, hyperedge_index2

    augmentation = None

    for i in range(1, epochs + 1):
        trans.train()
        model.train()

        optimizer_trans.zero_grad()

        loss = train_dis(trans, optimizer_trans, discriminator, optimizer_wd, model, updated_dataset1, updated_dataset2, augmentation)

        loss_feature = train_feature_recon(trans, optimizer_trans, model, updated_dataset1, updated_dataset2, [recon_model1, recon_model2], [optimizer_recon1, optimizer_recon2], augmentation)

        loss = (1-alpha) * loss + alpha * loss_feature
        loss.backward()
        optimizer_trans.step()

        model.eval()
        trans.eval()

        n_embd1 = model(features1, hyperedge_index1, num_nodes1, num_edges1)
        n_embd2 = model(features2, hyperedge_index2, num_nodes2, num_edges2)

        n_embd2 = trans(n_embd2)

        n_embd1[0] = 0
        n_embd2[0] = 0

        norm_1 = torch.nn.functional.normalize(n_embd1, dim=1, p=2)
        norm_2 = torch.nn.functional.normalize(n_embd2, dim=1, p=2)

        cossim = torch.mm(norm_1, norm_2.transpose(0,1))

        if augmentation_t > 0:
            # conduct augmentation here
            augment1, augment2 = topo_augment(hyperedge_index1, num_edges1, Dn1, hyperedge_index2, num_edges2, Dn2, cossim, augmentation_t)

            updated_hyperedge_index1, Dn1_, weights1, eweights1 = augment1
            updated_hyperedge_index2, Dn2_, weights2, eweights2 = augment2

            updated_num_edges1 = num_edges1 + num_edges2
            updated_num_edges2 = num_edges2 + num_edges1

            updated_dataset1 = num_nodes1, updated_num_edges1, features1, updated_hyperedge_index1
            updated_dataset2 = num_nodes2, updated_num_edges2, features2, updated_hyperedge_index2

            augmentation1 = (Dn1_, weights1, eweights1)
            augmentation2 = (Dn2_, weights2, eweights2)
            augmentation = (augmentation1, augmentation2)

    numpy_emb1 = n_embd1.detach().numpy()
    numpy_emb2 = n_embd2.detach().numpy()

    return numpy_emb1, numpy_emb2
    

# train discriminator
def train_dis(trans, optimizer_trans, discriminator, optimizer_d, model, dataset1, dataset2, augmentation, lambda_gp=10, batch_d_per_iter=5, batch_size_align=512):

    num_nodes1, num_edges1, features1, hyperedge_index1 = dataset1
    num_nodes2, num_edges2, features2, hyperedge_index2 = dataset2

    if augmentation!= None:
        augmentation1, augmentation2 = augmentation
    else:
        augmentation1 = None
        augmentation2 = None

    embd1 = model(features1, hyperedge_index1, num_nodes1, num_edges1, augmentation1)

    node_embd2 = model(features2, hyperedge_index2, num_nodes2, num_edges2, augmentation2)
    embd2 = trans(node_embd2)

    trans.train()
    discriminator.train()
    model.train()

    for j in range(batch_d_per_iter):
        w1 = discriminator(embd1)
        w2 = discriminator(embd2)

        anchor1 = w1.view(-1).argsort(descending=True)[: embd1.size(0)]
        anchor2 = w2.view(-1).argsort(descending=False)[: embd2.size(0)]

        embd1_anchor = embd1[anchor1, :].clone().detach()
        embd2_anchor = embd2[anchor2, :].clone().detach()
        optimizer_d.zero_grad()
        loss = -torch.mean(discriminator(embd1_anchor)) + torch.mean(discriminator(embd2_anchor))
        loss.backward()
        optimizer_d.step()
        for p in discriminator.parameters():
            p.data.clamp_(-0.1, 0.1)

    w1 = discriminator(embd1)
    w2 = discriminator(embd2)

    anchor1 = w1.view(-1).argsort(descending=True)[: embd1.size(0)]
    anchor2 = w2.view(-1).argsort(descending=False)[: embd2.size(0)]
	
    embd1_anchor = embd1[anchor1, :]
    embd2_anchor = embd2[anchor2, :]

    loss = -torch.mean(discriminator(embd1_anchor))
    return loss


# train reconstruction functions
def train_feature_recon(trans, optimizer_trans, model, dataset1, dataset2, recon_models, optimizer_recons, augmentation, batch_r_per_iter=10):
	
    recon_model1, recon_model2 = recon_models
    optimizer_recon1, optimizer_recon2 = optimizer_recons

    num_nodes1, num_edges1, features1, hyperedge_index1 = dataset1
    num_nodes2, num_edges2, features2, hyperedge_index2 = dataset2

    if augmentation!= None:
        augmentation1, augmentation2 = augmentation
    else:
        augmentation1 = None
        augmentation2 = None


    embd1 = model(features1, hyperedge_index1, num_nodes1, num_edges1, augmentation1)

    node_embd2 = model(features2, hyperedge_index2, num_nodes2, num_edges2, augmentation2)
    embd2 = trans(node_embd2)
	
    recon_model1.train()
    recon_model2.train()

    trans.train()
    model.train()

	
    embd1_copy = embd1.clone().detach()	
    embd2_copy = embd2.clone().detach()

    for t in range(batch_r_per_iter):
	    optimizer_recon1.zero_grad()
	    loss = feature_reconstruct_loss(embd1_copy, features1, recon_model1)
	    loss.backward()
	    optimizer_recon1.step()

    for t in range(batch_r_per_iter):
	    optimizer_recon2.zero_grad()
	    loss = feature_reconstruct_loss(embd2_copy, features2, recon_model2)
	    loss.backward()
	    optimizer_recon2.step()

    loss = 0.5 * feature_reconstruct_loss(embd1, features1, recon_model1) + 0.5 * feature_reconstruct_loss(embd2, features2, recon_model2)

    return loss

def topo_augment(hyperedge_index1, num_edges1, ori_Dn1, hyperedge_index2, num_edges2, ori_Dn2, cossim, t):

    Dn1 = torch.zeros(ori_Dn1.shape)
    Dn2 = torch.zeros(ori_Dn2.shape)

    _, top_idx1 = torch.topk(cossim, t, dim=1)
    mask = torch.zeros_like(cossim).scatter_(1, top_idx1, 1)
    sim1 = cossim * mask
    row_sums1 = sim1.abs().sum(dim=1, keepdim=True)
    norm_sim1 = sim1 / row_sums1 

    cossim_T = torch.transpose(cossim, 0, 1)
    _, top_idx2 = torch.topk(cossim_T, t, dim=1)
    mask = torch.zeros_like(cossim_T).scatter_(1, top_idx2, 1)
    sim2 = cossim_T * mask
    row_sums2 = sim2.abs().sum(dim=1, keepdim=True)
    norm_sim2 = sim2 / row_sums2

    
    count1 = num_edges1
    augment_node1 = []
    augment_edge1 = []
    weights1 = []
    eweights1 = []

    for j in range(hyperedge_index2.shape[1]):
        node_id = hyperedge_index2[0][j]
        
        corresponding_nodes = [cor_node.item() for cor_node in top_idx2[node_id]]
        for c_node in corresponding_nodes:
            augment_node1.append(c_node)
            augment_edge1.append(count1 + hyperedge_index2[1][j])
            weights1.append(norm_sim2[node_id][c_node])
            eweights1.append(t * norm_sim2[node_id][c_node])
            Dn1[c_node] += norm_sim2[node_id][c_node]

    
    augment_hyperedge_index1 = torch.tensor([augment_node1, augment_edge1])
    updated_hyperedge_index1 = torch.cat((hyperedge_index1, augment_hyperedge_index1), dim=1)
    weights_ori1 = torch.ones(Dn1.shape[0] + hyperedge_index1.shape[1])
    weights1 = torch.tensor(weights1)
    eweights1 = torch.tensor(eweights1)
    updated_weights1 = torch.cat((weights_ori1, weights1))
    updated_eweights1 = torch.cat((weights_ori1, eweights1))
    Dn1 = Dn1 + ori_Dn1

    
    count2 = num_edges2
    augment_node2 = []
    augment_edge2 = []
    weights2 = []
    eweights2 = []

    for j in range(hyperedge_index1.shape[1]):
        node_id = hyperedge_index1[0][j]
        
        corresponding_nodes = [cor_node.item() for cor_node in top_idx1[node_id]]
        for c_node in corresponding_nodes:
            augment_node2.append(c_node)
            augment_edge2.append(count2 + hyperedge_index1[1][j])
            weights2.append(norm_sim1[node_id][c_node])
            eweights2.append(t * norm_sim1[node_id][c_node])
            Dn2[c_node] += norm_sim1[node_id][c_node]

    augment_hyperedge_index2 = torch.tensor([augment_node2, augment_edge2])
    updated_hyperedge_index2 = torch.cat((hyperedge_index2, augment_hyperedge_index2), dim=1)
    weights_ori2 = torch.ones(Dn2.shape[0] + hyperedge_index2.shape[1])
    weights2 = torch.tensor(weights2)
    eweights2 = torch.tensor(eweights2)
    updated_weights2 = torch.cat((weights_ori2, weights2))
    updated_eweights2 = torch.cat((weights_ori2, eweights2))
    Dn2 = Dn2 + ori_Dn2

    augment1 = (updated_hyperedge_index1, Dn1, updated_weights1, updated_eweights1)
    augment2 = (updated_hyperedge_index2, Dn2, updated_weights2, updated_eweights2)
    
    return augment1, augment2
    