#!/usr/bin/python
# -*- coding: utf-8 -*-

import scipy
import argparse, logging
import numpy as np
import hf2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time

import graph


def parse_args():
	'''
	Parses the hyperfeat arguments.
	'''
	parser = argparse.ArgumentParser(description="Run HyperFeat.")

	parser.add_argument('--input', nargs='?', default='barbell',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/barbell.emb',
	                    help='Embeddings path')

	parser.add_argument('--hf', type=int, default=1,
	                    help='1: hyperfeat, 0: struc2vec')

	parser.add_argument('--dimensions', type=int, default=32,
	                    help='Number of dimensions. Default is 32.')

	parser.add_argument('--walk-length', type=int, default=50,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--until-layer', type=int, default=3,
                    	help='Calculation until the layer.')

	parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers. Default is 8.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	parser.add_argument('--OPT1', default=True, type=bool,
                      help='optimization 1')
	return parser.parse_args()

def read_hypergraph():
	
	G, incidence_size = graph.load_hyperedge('graph/{}.txt'.format(args.input))
	
	return G, incidence_size

def clique_expansion():
	f = open('graph/{}.txt'.format(args.input), 'r')
	g = open('graph/{}.edgelist'.format(args.input), 'w')

	for line in f:
		lines = line.split(' ')
		for i in range(len(lines)):
			for j in range(i+1, len(lines)):
				g.write(lines[i].replace('\n', '') + ' ' + lines[j].replace('\n', '') + '\n')
	f.close()
	g.close()

def read_graph():
	'''
	Reads the input network.
	'''
	
	G = graph.load_edgelist('graph/{}.edgelist'.format(args.input), undirected=True)
	
	return G

def learn_embeddings():
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	
	walks = LineSentence('random_walks.txt')
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1, workers=4, epochs=args.iter)
	model.wv.save_word2vec_format(args.output)
	
	
	return

def exec_hf2vec(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''

	
	if args.hf:
		G, incidence_size = read_hypergraph()
		

	else:
		print('run here')
		clique_expansion()
		G = read_graph()
		incidence_size = None

	G = hf2vec.Graph(G, incidence_size)

	if(args.OPT1):
		
		degreeList, degreeList_ = G.preprocess_neighbors_with_bfs_compact()
	else:
		
		degreeList, degreeList_ = G.preprocess_neighbors_with_bfs()


	if args.hf:
		vertices, degreeList, distances = G.calc_distances_alternative(degreeList, degreeList_, compactDegree = args.OPT1)

	else:
		
		vertices, degreeList, distances = G.calc_distances(degreeList, compactDegree = args.OPT1)
	

		
	weights, alias_method_j, alias_method_q, graph_c, alias_method_j_c, alias_method_q_c = G.create_distances_network(vertices, degreeList, distances)
	amount_neighbors = G.preprocess_parameters_random_walk(weights, alias_method_j, alias_method_q)

	G.simulate_walks(graph_c,alias_method_j, alias_method_q,amount_neighbors, args.num_walks, args.walk_length)
	
	#print(distances)
	return G

def main(args):
	
	G = exec_hf2vec(args)
	
	learn_embeddings()
	
	'''
	G, incidence_size = read_hypergraph()
	print(incidence_size)
	print('load hypergraph successfully')
	'''

if __name__ == "__main__":
	args = parse_args()
	main(args)

