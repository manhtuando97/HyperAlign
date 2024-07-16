# -*- coding: utf-8 -*-
from time import time
from collections import deque
import numpy as np
import math,logging
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from hyperFeat.utils import *
import os

limiteDist = 20

def getDegreeListsVertices(g,incidence_size,vertices,calcUntilLayer):
    degreeList = {}
    degreeList_ = {}

    for v in vertices:
        degreeList[v] = getDegreeLists(g,v,calcUntilLayer)

        if incidence_size != None:
            degreeList_[v] = getDegreeLists_(g, v, incidence_size, calcUntilLayer)

    return degreeList, degreeList_

def getCompactDegreeListsVertices(g,incidence_size,vertices,maxDegree,calcUntilLayer):
    degreeList = {}
    degreeList_ = {}

    for v in vertices:
        degreeList[v] = getCompactDegreeLists(g,v,maxDegree,calcUntilLayer)
        #degreeList_[v] = getCompactDegreeLists_(g,v,incidence_size,maxDegree,calcUntilLayer)
        if incidence_size != None:
            degreeList_[v] = getDegreeLists_(g, v, incidence_size, calcUntilLayer)

    return degreeList, degreeList_


def getCompactDegreeLists_(g, root, incidence_size,maxDegree,calcUntilLayer,max_size=10):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    l = {}
    
    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        d = len(g[vertex])
        if(d not in l):
            l[d] = 0
        l[d] += 1

        for v in g[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1    

        if(timeToDepthIncrease == 0):

            list_d = []
            for degree,freq in l.items():
                list_d.append((degree,freq))
            list_d.sort(key=lambda x: x[0])

            #l_sorted = np.sort(list_d)

            lp = {}
            for size in range(2, max_size+1):
                lp[size] = [incidence_size[neighbor][size] for neighbor in list_d]
            # listas[depth] should be a dictionary, each each is a size, value is list of # incident hyperedges of that size of nodes in those depth
            listas[depth] = lp

            l = {}

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()

    return listas

def getCompactDegreeLists(g, root, maxDegree,calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    l = {}
    
    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        d = len(g[vertex])
        if(d not in l):
            l[d] = 0
        l[d] += 1

        for v in g[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1    

        if(timeToDepthIncrease == 0):

            list_d = []
            for degree,freq in l.items():
                list_d.append((degree,freq))
            list_d.sort(key=lambda x: x[0])
            listas[depth] = np.array(list_d,dtype=np.int32)

            l = {}

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()
    

    return listas

def getDegreeLists_(g, root, incidence_size, calcUntilLayer, max_size = 10):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    

    l = deque()
    
    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        l.append(vertex)

        for v in g[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1    

        if(timeToDepthIncrease == 0):

            lp = np.array(l,dtype='int')
            l_sorted = np.sort(lp)

            lp = {}
            for size in range(2, max_size+1):
                lp[size] = [incidence_size[neighbor][size] for neighbor in l_sorted]
            # listas[depth] should be a dictionary, each each is a size, value is list of # incident hyperedges of that size of nodes in those depth
            listas[depth] = lp
            l = deque()

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()

    #print('in getDegreeList listas')
    #print(listas)
    return listas

def getDegreeLists(g, root, calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    

    l = deque()
    
    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        l.append(len(g[vertex]))

        for v in g[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1    

        if(timeToDepthIncrease == 0):

            lp = np.array(l,dtype='int')
            lp = np.sort(lp)
            listas[depth] = lp
            l = deque()

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()

    #print('in getDegreeList listas')
    #print(listas)
    return listas

def cost(a,b):
    ep = 0.5
    m = max(a,b) + ep
    mi = min(a,b) + ep
    return ((m/mi) - 1)

def cost_min(a,b):
    ep = 0.5
    m = max(a[0],b[0]) + ep
    mi = min(a[0],b[0]) + ep
    return ((m/mi) - 1) * min(a[1],b[1])


def cost_max(a,b):
    print(a)
    print(b)
    ep = 0.5
    m = max(a[0],b[0]) + ep
    mi = min(a[0],b[0]) + ep
    return ((m/mi) - 1) * max(a[1],b[1])

def preprocess_degreeLists():

    degreeList = restoreVariableFromDisk('degreeList')

    dList = {}
    dFrequency = {}
    for v,layers in degreeList.items():
        dFrequency[v] = {}
        for layer,degreeListLayer in layers.items():
            dFrequency[v][layer] = {}
            for degree in degreeListLayer:
                if(degree not in dFrequency[v][layer]):
                    dFrequency[v][layer][degree] = 0
                dFrequency[v][layer][degree] += 1
    for v,layers in dFrequency.items():
        dList[v] = {}
        for layer,frequencyList in layers.items():
            list_d = []
            for degree,freq in frequencyList.items():
                list_d.append((degree,freq))
            list_d.sort(key=lambda x: x[0])
            dList[v][layer] = np.array(list_d,dtype='float')

    #saveVariableOnDisk(dList,'compactDegreeList')

    return dList

def verifyDegrees(degrees,degree_v_root,degree_a,degree_b):

    if(degree_b == -1):
        degree_now = degree_a
    elif(degree_a == -1):
        degree_now = degree_b
    elif(abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root)):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now 

def get_vertices(v,degree_v,degrees,a_vertices):
    #a_vertices_selected = 2 * math.log(a_vertices,2)
    a_vertices_selected =  math.log(a_vertices,2)
    vertices = deque()

    try:
        c_v = 0  

        for v2 in degrees[degree_v]['vertices']:
            if(v != v2):
                vertices.append(v2)
                c_v += 1
                if(c_v > a_vertices_selected):
                    raise StopIteration

        if('before' not in degrees[degree_v]):
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        if('after' not in degrees[degree_v]):
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']
        if(degree_b == -1 and degree_a == -1):
            raise StopIteration
        degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

        while True:
            for v2 in degrees[degree_now]['vertices']:
                if(v != v2):
                    vertices.append(v2)
                    c_v += 1
                    if(c_v > a_vertices_selected):
                        raise StopIteration

            if(degree_now == degree_b):
                if('before' not in degrees[degree_b]):
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if('after' not in degrees[degree_a]):
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']
            
            if(degree_b == -1 and degree_a == -1):
                raise StopIteration

            degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

    except StopIteration:
        return list(vertices)

    return list(vertices)


def splitDegreeList(G, degreeList, degrees,compactDegree):

    '''
    if(compactDegree):
        #degreeList = restoreVariableFromDisk('compactDegreeList')
        degreeList = exec_bfs_compact(self.G,self.workers,self.calcUntilLayer)
    else:
        #degreeList = restoreVariableFromDisk('degreeList')
        degreeList = exec_bfs(self.G,self.workers,self.calcUntilLayer)
    

    #degreeList = preprocess_degreeLists()
    '''

    #degrees = restoreVariableFromDisk('degrees_vector')

    degreeListsSelected = {}
    vertices = {}
    a_vertices = len(G)

    c = G.keys()

    for v in c:
        nbs = get_vertices(v,len(G[v]),degrees,a_vertices)
        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            degreeListsSelected[n] = degreeList[n]

    #saveVariableOnDisk(vertices,'split-vertices-'+str(part))
    #saveVariableOnDisk(degreeListsSelected,'split-degreeList-'+str(part))
    
    return vertices, degreeListsSelected


def calc_distances_alternative(vertices, degreeList, max_size = 5, compactDegree = False):

    #vertices = restoreVariableFromDisk('split-vertices-'+str(part))
    #degreeList = restoreVariableFromDisk('split-degreeList-'+str(part))

    distances = {}
    if compactDegree:
        dist_func = cost_max
    else:
        dist_func = cost
    
    
    for v1,nbs in vertices.items():
        lists_v1 = degreeList[v1]
        
        for v2 in nbs:
            t00 = time()
            lists_v2 = degreeList[v2]

            max_layer_ = min(len(lists_v1),len(lists_v2))
            max_layer = min(2, max_layer_)
            distances[v1,v2] = {}

            for layer in range(0,max_layer):
                distance = 0
                
                for size in range(2, max_size+1):
                
                    dist, path = fastdtw(lists_v1[layer][size],lists_v2[layer][size],radius=1,dist=dist_func)
                    distance += dist

                distances[v1,v2][layer] = distance

            t11 = time()


    preprocess_consolides_distances(distances)
    #saveVariableOnDisk(distances,'distances-'+str(part))
    return distances

def calc_distances(vertices, degreeList, compactDegree = False):

    #vertices = restoreVariableFromDisk('split-vertices-'+str(part))
    #degreeList = restoreVariableFromDisk('split-degreeList-'+str(part))

    distances = {}
    if compactDegree:
        dist_func = cost_max
    else:
        dist_func = cost
    
    
    for v1,nbs in vertices.items():
        lists_v1 = degreeList[v1]
        # lists_v1 is a dictionary

        for v2 in nbs:
            t00 = time()
            lists_v2 = degreeList[v2]

            max_layer_ = min(len(lists_v1),len(lists_v2))

            max_layer = min(2, max_layer_)
            distances[v1,v2] = {}
            

            for layer in range(0,max_layer):
                
                dist, path = fastdtw(lists_v1[layer][0],lists_v2[layer][0],radius=1,dist=dist_func)

                distances[v1,v2][layer] = dist

            t11 = time()


    preprocess_consolides_distances(distances)
    return distances

def calc_distances_all(vertices,list_vertices,degreeList,compactDegree = False):

    distances = {}
    cont = 0

    if compactDegree:
        dist_func = cost_max
    else:
        dist_func = cost

    for v1 in vertices:
        lists_v1 = degreeList[v1]

        for v2 in list_vertices[cont]:
            lists_v2 = degreeList[v2]
            
            max_layer = min(len(lists_v1),len(lists_v2))
            distances[v1,v2] = {}

            for layer in range(0,max_layer):
                #t0 = time()
                
                #dist, path = fastdtw(list(lists_v1[layer][0]),list(lists_v2[layer][0]),radius=1,dist=dist_func)
                dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
                #t1 = time()
                distances[v1,v2][layer] = dist
                

        cont += 1

    preprocess_consolides_distances(distances)
    #saveVariableOnDisk(distances,'distances-'+str(part))
    print('calc_distances_all in algorithms_distances.py')
    print(distances)
    return distances


def selectVertices(layer,fractionCalcDists):
    previousLayer = layer - 1

    distances = restoreVariableFromDisk('distances')

    threshold = calcThresholdDistance(previousLayer,distances,fractionCalcDists)


    vertices_selected = deque()

    for vertices,layers in distances.items():
        if(previousLayer not in layers):
            continue
        if(layers[previousLayer] <= threshold):
            vertices_selected.append(vertices)

    distances = {}

    return vertices_selected


def preprocess_consolides_distances(distances, startLayer = 1):

    for vertices,layers in distances.items():
        keys_layers = sorted(layers.keys())
        startLayer = min(len(keys_layers),startLayer)
        for layer in range(0,startLayer):
            keys_layers.pop(0)


        for layer in keys_layers:
            layers[layer] += layers[layer - 1]


def exec_bfs_compact(G,incidence_size,workers,calcUntilLayer):

    futures = {}
    degreeList = {}

    t0 = time()
    vertices = G.keys()
    parts = workers
    chunks = partition(vertices,parts)

    maxDegree = 0
    for v in vertices:
        if(len(G[v]) > maxDegree):
            maxDegree = len(G[v])

    '''
    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            job = executor.submit(getCompactDegreeListsVertices,G,c,maxDegree,calcUntilLayer)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)
    '''
    degreeList, degreeList_ = getCompactDegreeListsVertices(G,incidence_size,vertices,maxDegree,calcUntilLayer)

    #saveVariableOnDisk(degreeList,'compactDegreeList')
    t1 = time()


    return degreeList, degreeList_

def exec_bfs(G,incidence_size,workers,calcUntilLayer):

    futures = {}
    degreeList = {}

    t0 = time()
    vertices = G.keys()
    parts = workers
    #chunks = partition(vertices,parts)
    '''
    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            job = executor.submit(getDegreeListsVertices,G,c,calcUntilLayer)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)
    '''
    #print('exec bfs vertices')
    #print(G)
    #print(vertices)
    degreeList, degreeList_ = getDegreeListsVertices(G,incidence_size,vertices,calcUntilLayer)

    #saveVariableOnDisk(degreeList,'degreeList')
    t1 = time()


    return degreeList, degreeList_


def generate_distances_network_part1(vertices, degreeList, distances, workers):
    parts = workers
    weights_distances = {}
    '''
    for part in range(1,parts + 1):    
        
        distances = restoreVariableFromDisk('distances-'+str(part))
        
        for vertices,layers in distances.items():
            for layer,distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in weights_distances):
                    weights_distances[layer] = {}
                weights_distances[layer][vx,vy] = distance

    '''
    #distances = calc_distances(vertices, degreeList)

    for vertices,layers in distances.items():
            for layer,distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in weights_distances):
                    weights_distances[layer] = {}
                weights_distances[layer][vx,vy] = distance

    weights_distances_layer = dict()
    for layer,values in weights_distances.items():
        weights_distances_layer[layer] = values
        #saveVariableOnDisk(values,'weights_distances-layer-'+str(layer))
    return weights_distances_layer

def generate_distances_network_part2(vertices, degreeList, workers):
    parts = workers
    graphs = {}

    distances = calc_distances(vertices, degreeList)

    '''
    for part in range(1,parts + 1):

        distances = restoreVariableFromDisk('distances-'+str(part))

        for vertices,layers in distances.items():
            for layer,distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in graphs):
                    graphs[layer] = {}
                if(vx not in graphs[layer]):
                   graphs[layer][vx] = [] 
                if(vy not in graphs[layer]):
                   graphs[layer][vy] = [] 
                graphs[layer][vx].append(vy)
                graphs[layer][vy].append(vx)
    '''

    for vertices,layers in distances.items():
        for layer,distance in layers.items():
            vx = vertices[0]
            vy = vertices[1]
            if(layer not in graphs):
                graphs[layer] = {}
            if(vx not in graphs[layer]):
                graphs[layer][vx] = [] 
            if(vy not in graphs[layer]):
                graphs[layer][vy] = [] 
            graphs[layer][vx].append(vy)
            graphs[layer][vy].append(vx)

    layer_graphs = dict()
    for layer,values in graphs.items():
        layer_graphs[layer] = values
        #saveVariableOnDisk(values,'graphs-layer-'+str(layer))

    return layer_graphs

def generate_distances_network_part3(vertices, degreeList, distances):

    layer = 0

    weights_distances_layer = generate_distances_network_part1(vertices, degreeList, distances, 4)
    layer_graphs = generate_distances_network_part2(vertices, degreeList, 4)
    #while(isPickle('graphs-layer-'+str(layer))):
    alias_method_j = {}
    alias_method_q = {}
    weights = {}
    for layer in layer_graphs.keys():
        #graphs = restoreVariableFromDisk('graphs-layer-'+str(layer))
        graphs = layer_graphs[layer]
        #weights_distances = restoreVariableFromDisk('weights_distances-layer-'+str(layer))
        weights_distances = weights_distances_layer[layer]

        alias_method_j[layer] = {}
        alias_method_q[layer] = {}
        weights[layer] = {}
    
        for v,neighbors in graphs.items():
            e_list = deque()
            sum_w = 0.0


            for n in neighbors:
                if (v,n) in weights_distances:
                    wd = weights_distances[v,n]
                else:
                    wd = weights_distances[n,v]
                w = np.exp(-float(wd))
                e_list.append(w)
                sum_w += w

            e_list = [x / sum_w for x in e_list]
            weights[layer][v] = e_list
            J, q = alias_setup(e_list)
            alias_method_j[layer][v] = J
            alias_method_q[layer][v] = q

        #saveVariableOnDisk(weights,'distances_nets_weights-layer-'+str(layer))
        #saveVariableOnDisk(alias_method_j,'alias_method_j-layer-'+str(layer))
        #saveVariableOnDisk(alias_method_q,'alias_method_q-layer-'+str(layer))
        layer += 1

    return weights, alias_method_j, alias_method_q, layer_graphs


def generate_distances_network_part4(layer_graphs):
    graphs_c = {}
    layer = 0
    #while(isPickle('graphs-layer-'+str(layer))):
    for layer in layer_graphs.keys():
        
        #graphs = restoreVariableFromDisk('graphs-layer-'+str(layer))
        graphs = layer_graphs[layer]
        graphs_c[layer] = graphs
        
        #layer += 1


    #saveVariableOnDisk(graphs_c,'distances_nets_graphs')
    return graphs_c

def generate_distances_network_part5(alias_method_j):
    alias_method_j_c = {}
    layer = 0
    #while(isPickle('alias_method_j-layer-'+str(layer))):
    for layer in alias_method_j.keys():       

        #alias_method_j = restoreVariableFromDisk('alias_method_j-layer-'+str(layer))
        alias_method_j_c[layer] = alias_method_j[layer]
        #layer += 1
    #saveVariableOnDisk(alias_method_j_c,'nets_weights_alias_method_j')

    return alias_method_j_c

def generate_distances_network_part6(alias_method_q):
    alias_method_q_c = {}
    layer = 0
    #while(isPickle('alias_method_q-layer-'+str(layer))):
    for layer in alias_method_q.keys():
        
        #alias_method_q = restoreVariableFromDisk('alias_method_q-layer-'+str(layer))
        alias_method_q_c[layer] = alias_method_q[layer]
        #layer += 1

    #saveVariableOnDisk(alias_method_q_c,'nets_weights_alias_method_q')

    return alias_method_q_c

def generate_distances_network(vertices, degreeList, distances):
    t0 = time()
    workers = 4
    '''
    os.system("rm "+returnPathStruc2vec()+"/../pickles/weights_distances-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part1,workers)
        job.result()
    t1 = time()
    t = t1-t0
    
    t0 = time()
    os.system("rm "+returnPathStruc2vec()+"/../pickles/graphs-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part2,workers)
        job.result()
    t1 = time()
    t = t1-t0

    t0 = time()
    os.system("rm "+returnPathStruc2vec()+"/../pickles/distances_nets_weights-layer-*.pickle")
    os.system("rm "+returnPathStruc2vec()+"/../pickles/alias_method_j-layer-*.pickle")
    os.system("rm "+returnPathStruc2vec()+"/../pickles/alias_method_q-layer-*.pickle")

    
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part3)
        job.result()
    t1 = time()
    t = t1-t0
    
    
    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part4)
        job.result()
    t1 = time()
    t = t1-t0

    
    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part5)
        job.result()
    t1 = time()
    t = t1-t0

    t0 = time()
    
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part6)
        job.result()
    '''
    #generate_distances_network_part1(vertices, degreeList, 4)
    #layer_graphs = generate_distances_network_part2(vertices, degreeList, 4)
    weights, alias_method_j, alias_method_q, layer_graphs = generate_distances_network_part3(vertices, degreeList, distances)
    graph_c = generate_distances_network_part4(layer_graphs)
    alias_method_j_c = generate_distances_network_part5(alias_method_j)
    alias_method_q_c = generate_distances_network_part6(alias_method_q)
    t1 = time()
    t = t1-t0
 
    return weights, alias_method_j, alias_method_q, graph_c, alias_method_j_c, alias_method_q_c


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q
