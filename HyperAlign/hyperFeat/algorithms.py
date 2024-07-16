# -*- coding: utf-8 -*-
from time import time
from collections import deque
import numpy as np
import math,random,logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict

from hyperFeat.utils import *


def generate_parameters_random_walk(weights_, workers):

    sum_weights = {}
    amount_edges = {}
    
    layer = 0
    for layer in weights_.keys():
        weights = weights_[layer]
    
        for k,list_weights in weights.items():
            if(layer not in sum_weights):
                sum_weights[layer] = 0
            if(layer not in amount_edges):
                amount_edges[layer] = 0

            for w in list_weights:
                sum_weights[layer] += w
                amount_edges[layer] += 1
        

    average_weight = {}
    for layer in sum_weights.keys():
        average_weight[layer] = sum_weights[layer] / amount_edges[layer]


    amount_neighbours = {}

    layer = 0

    for layer in weights_.keys():
                
        weights = weights_[layer]

        amount_neighbours[layer] = {}

        for k,list_weights in weights.items():
            cont_neighbours = 0
            for w in list_weights:
                if(w > average_weight[layer]):
                    cont_neighbours += 1
            amount_neighbours[layer][k] = cont_neighbours

    return amount_neighbours

def chooseNeighbor(v,graphs,alias_method_j,alias_method_q,layer):
    
    v_list = graphs[layer][v]

    idx = alias_draw(alias_method_j[layer][v],alias_method_q[layer][v])
    v = v_list[idx]

    return v


def exec_random_walk(graphs,alias_method_j,alias_method_q,v,walk_length,amount_neighbours):
    
    original_v = v
    t0 = time()
    initialLayer = 0
    layer = initialLayer


    path = deque()
    path.append(v)

    while len(path) < walk_length:
        r = random.random()

        if(r < 0.3):
                v = chooseNeighbor(v,graphs,alias_method_j,alias_method_q,layer)
                path.append(v)

        else:
            r = random.random()
            
            limiar_moveup = prob_moveup(amount_neighbours[layer][v])
            if(r > limiar_moveup):
                if(layer > initialLayer):
                    layer = layer - 1           
            else:
                if((layer + 1) in graphs and v in graphs[layer + 1]):
                    layer = layer + 1

    t1 = time()

    return path


def exec_ramdom_walks_for_chunck(vertices,graphs,alias_method_j,alias_method_q,walk_length,amount_neighbours):
    walks = deque()
    for v in vertices:
        walks.append(exec_random_walk(graphs,alias_method_j,alias_method_q,v,walk_length,amount_neighbours))
    return walks

def generate_random_walks_large_graphs(name, graph_c, alias_method_j, alias_method_q,amount_neighbors,num_walks,walk_length,workers,vertices):


    t0 = time()
    
    walks = deque()
    initialLayer = 0

    parts = workers

    with ProcessPoolExecutor(max_workers=workers) as executor:

        for walk_iter in range(num_walks):
            random.shuffle(vertices)
            
            walk = exec_ramdom_walks_for_chunck(vertices,graphs,alias_method_j,alias_method_q,walk_length,amount_neighbors)
            walks.extend(walk)
            



    t1 = time()
    save_random_walks(name, walks)

def generate_random_walks(name, graphs,alias_method_j, alias_method_q,amount_neighbours,num_walks,walk_length,workers,vertices):

    t0 = time()
    
    walks = deque()
    initialLayer = 0

    if(workers > num_walks):
        workers = num_walks
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for walk_iter in range(num_walks):
            random.shuffle(vertices)
            job = executor.submit(exec_ramdom_walks_for_chunck,vertices,graphs,alias_method_j,alias_method_q,walk_length,amount_neighbours)
            futures[job] = walk_iter
            
        for job in as_completed(futures):
            walk = job.result()
            r = futures[job]
            walks.extend(walk)
            del futures[job]

    t1 = time()
    
    save_random_walks(name, walks)

def save_random_walks(name, walks):
    with open('random walks/random_walks-{}.txt'.format(name), 'w') as file:
        for walk in walks:
            line = ''
            for v in walk:
                line += str(v)+' '
            line += '\n'
            file.write(line)
    return

def prob_moveup(amount_neighbours):
    x = math.log(amount_neighbours + math.e)
    p = (x / ( x + 1))
    return p



def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
