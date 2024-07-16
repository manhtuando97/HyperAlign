import scipy
import argparse, logging
import numpy as np

import hyperFeat.hf2vec as hf2vec
from hyperFeat.hf2vec import *
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time

from hyperFeat.graph import *
from hyperFeat.graph import Graph


def read_hypergraph(input_dir, name):
	
	G, incidence_size = load_hyperedge('{}/{}.txt'.format(input_dir, name))
	
	return G, incidence_size

def clique_expansion(input_dir, name):
	f = open('{}/{}.txt'.format(input_dir, name), 'r')
	g = open('{}/{}.edgelist'.format(input_dir, name), 'w')

	for line in f:
		lines = line.split(' ')
		for i in range(len(lines)):
			for j in range(i+1, len(lines)):
				g.write(lines[i].replace('\n', '') + ' ' + lines[j].replace('\n', '') + '\n')
	f.close()
	g.close()

def read_graph(input_dir, name):
	'''
	Reads the input network.
	'''
	G = load_edgelist('{}/{}.edgelist'.format(input_dir, name), undirected=True)
	
	return G

def learn_embeddings(name, dimension, window_size, iteration, feat_output):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	
	walks = LineSentence('random walks/random_walks-{}.txt'.format(name))
	model = Word2Vec(walks, vector_size=dimension, window=window_size, min_count=0, hs=1, sg=1, workers=4, epochs=iteration)
	model.wv.save_word2vec_format(feat_output + '/' + name + '.emb')
	
	
	return

def exec_hf2vec(input_dir, name, hf, config, opt, num_walks, walk_length):
	'''
	Pipeline for representational learning for all nodes in a hypergraph.
	'''

	if hf == 1:
		G, incidence_size = read_hypergraph(input_dir, name)
		

	elif (hf == 0) or (config == 1):
		
		clique_expansion(input_dir, name)
		G = read_graph(input_dir, name)
		incidence_size = None

	else:
		print('Invalid Configuration.')
		return None

	G = hf2vec.Graph(G, incidence_size)
	

	if(opt):
		
		degreeList, degreeList_ = G.preprocess_neighbors_with_bfs_compact()
	else:
		
		degreeList, degreeList_ = G.preprocess_neighbors_with_bfs()


	if hf == 1:
		vertices, degreeList, distances = G.calc_distances_alternative(degreeList, degreeList_, compactDegree = opt)

	elif (hf == 0) or (config == 1):
		
		vertices, degreeList, distances = G.calc_distances(degreeList, compactDegree = opt)
	

		
	weights, alias_method_j, alias_method_q, graph_c, alias_method_j_c, alias_method_q_c = G.create_distances_network(vertices, degreeList, distances)
	amount_neighbors = G.preprocess_parameters_random_walk(weights, alias_method_j, alias_method_q)

	G.simulate_walks(name, graph_c,alias_method_j, alias_method_q,amount_neighbors, num_walks, walk_length)
	
	return G

def hyperfeat(name, input_dir, hf, config, opt, num_walks, walk_length, dimension, window_size, iteration, feat_output):
	# run hyperfeat on 1 dataset

	G = exec_hf2vec(input_dir, name, hf, config, opt, num_walks, walk_length)
	learn_embeddings(name, dimension, window_size, iteration, feat_output)