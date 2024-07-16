import scipy

import numpy as np
import torch
import itertools
seed = 1
torch.manual_seed(seed)
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter_add
import time
import argparse
import pickle


from utils import  feature_reconstruct_loss, load_data
from gan import Discriminator, ReconDNN, transformation, notrans

# hyperfeat
from hyperfeat import hyperfeat

# hypercl
from models import HyperEncoder, hyperCL
from utils import drop_features, drop_incidence
from hypercl import train_cl

# hyperaug
from hyperaug import train_adv



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# general
	parser.add_argument('--dataset1', type=str, default='contact-high-school1')
	parser.add_argument('--dataset2', type=str, default='contact-high-school2')
	parser.add_argument('--config', type=int, default=0, help='0: full-fledged, 1: Hyper-S, 2: Hyper-WC, 3: Hyper-WA, 4: Hyper-WAC')
	parser.add_argument('--input', nargs='?', default='dataset', help='Input directory')
	parser.add_argument('--output', nargs='?', default='output', help='Output directory')

	
	# for hyperfeat
	
	parser.add_argument('--feat_output', nargs='?', default='dataset/features', help='Output directory for feature')
	parser.add_argument('--hf', type=int, default=1, help='1: hyperfeat, 0: struc2vec')
	parser.add_argument('--input_dimensions', type=int, default=32, help='Number of dimensions. Default is 32.')
	parser.add_argument('--walk_length', type=int, default=50, help='Length of walk per source. Default is 80.')
	parser.add_argument('--num_walks', type=int, default=10, help='Number of walks per source. Default is 10.')
	parser.add_argument('--window_size', type=int, default=10, help='Context size for optimization. Default is 10.')
	parser.add_argument('--iter', default=5, type=int, help='Number of epochs in SGD')
	parser.add_argument('--OPT1', default=True, type=bool, help='optimization in constructing level graphs')
	

	# for hyperCL
	
	parser.add_argument('--hid_dim', type=int, default=32)
	parser.add_argument('--num_layers', type=int, default=2)
	parser.add_argument('--lr', type=float, default=5.0e-04)
	parser.add_argument('--weight_decay', type=float, default=1.0e-05)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--drop_feature_rate', type=float, default=0.2)
	parser.add_argument('--drop_incidence_rate', type=float, default=0.2)
	parser.add_argument('--tau_n', type=float, default=0.5)
	

	# for hyperaug
	parser.add_argument('--transformer', type=int, default=1)
	parser.add_argument('--lr_gan', type=float, default=0.001)
	parser.add_argument('--lr_wd', type=float, default=0.01)
	parser.add_argument('--lr_recon', type=float, default=0.01)
	parser.add_argument('--alpha', type=float, default=0.01)
	parser.add_argument('--episode', type=int, default=10)
	parser.add_argument('--t', type=int, default=3)

	args = parser.parse_args()
	
	# HyperFeat module
	name1 = args.dataset1
	name2 = args.dataset2
	input_dir = args.input
	hf = args.hf
	config = args.config

	if config == 1:
		hf = 0

	config = args.config
	opt = args.OPT1
	num_walks = args.num_walks
	walk_length = args.walk_length

	dimension = args.input_dimensions
	window_size = args.window_size
	iteration = args.iter
	feat_output = args.feat_output

	hyperfeat(name1, input_dir, hf, config, opt, num_walks, walk_length, dimension, window_size, iteration, feat_output)
	hyperfeat(name2, input_dir, hf, config, opt, num_walks, walk_length, dimension, window_size, iteration, feat_output)


	# HyperCL module
	num_nodes1, num_edges1, features1, hyperedge_index1, node_list1 = load_data(args.input, args.feat_output, args.dataset1)
	num_nodes2, num_edges2, features2, hyperedge_index2, node_list2 = load_data(args.input, args.feat_output, args.dataset2)

	
	encoder = HyperEncoder(features1.shape[1], args.hid_dim, args.hid_dim, args.num_layers)
        
	model = hyperCL(encoder, args.hid_dim)

	if config != 2 and config != 4:
		optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

		for epoch in range(1, args.epochs + 1):
			train_cl(args, model, optimizer, num_nodes1, num_edges1, features1, hyperedge_index1, num_negs=None)

		for epoch in range(1, args.epochs + 1):
			train_cl(args, model, optimizer, num_nodes2, num_edges2, features2, hyperedge_index2, num_negs=None)
	
	hyperedge_weight1 = features1.new_ones(num_edges1)
	hyperedge_weight2 = features1.new_ones(num_edges2)
    
	Dn1 = scatter_add(hyperedge_weight1[hyperedge_index1[1]], hyperedge_index1[0], dim=0, dim_size=num_nodes1)
	Dn2 = scatter_add(hyperedge_weight2[hyperedge_index2[1]], hyperedge_index2[0], dim=0, dim_size=num_nodes2)


	# HyperAug module	
	dataset1 = num_nodes1, num_edges1, features1, hyperedge_index1, Dn1
	dataset2 = num_nodes2, num_edges2, features2, hyperedge_index2, Dn2

	if args.transformer == 1:
		trans = transformation(args.hid_dim, args.hid_dim, args.hid_dim)
	else:
		trans = notrans()
	
	optimizer_trans = torch.optim.Adam(itertools.chain(trans.parameters(), model.parameters()), lr=args.lr_gan, weight_decay=5e-4)

	aug_t = args.t
	if config == 3 or config == 4:
		aug_t = 0

	train_adv(model, trans, optimizer_trans, dataset1, dataset2, args.episode, args.alpha, args.lr_wd, args.lr_recon, aug_t)
	

	# Final node embeddings
	model.eval()
	n_embd1 = model(features1, hyperedge_index1, num_nodes1, num_edges1)
	n_embd2 = model(features2, hyperedge_index2, num_nodes2, num_edges2)

	n_embd2 = trans(n_embd2)

	f1 = open('{}/{}.emb'.format(args.output, args.dataset1), 'w')
	for node_id in node_list1:
		emb_node = n_embd1[node_id]
		emb_node_list = emb_node.tolist()
		feat_string = ' '.join(str(element) for element in emb_node_list)
		f1.write('{} {}\n'.format(str(node_id), feat_string))
	f1.close()

	f2 = open('{}/{}.emb'.format(args.output, args.dataset2), 'w')
	for node_id in node_list2:
		emb_node = n_embd2[node_id]
		emb_node_list = emb_node.tolist()
		feat_string = ' '.join(str(element) for element in emb_node_list)
		f2.write('{} {}\n'.format(str(node_id), feat_string))
	f2.close()


