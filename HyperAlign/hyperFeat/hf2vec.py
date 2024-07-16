# -*- coding: utf-8 -*-

import numpy as np
import random,sys,logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from time import time
from collections import deque

from hyperFeat.utils import *
from hyperFeat.algorithms import *
from hyperFeat.algorithms_distances import *
from hyperFeat.graph import *


class Graph():
	def __init__(self, g, incidence_size, untilLayer = None):

		self.G = g.gToDict()
	

		self.num_vertices = g.number_of_nodes()
		self.num_edges = g.number_of_edges()
		self.incidence_size = incidence_size
		self.workers = 4
		self.calcUntilLayer = untilLayer


	def preprocess_neighbors_with_bfs(self):
		'''
		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_bfs,self.G,self.workers,self.calcUntilLayer)
			
			job.result()
		'''
		#print('preprocess neighbors bfs')
		#print(self.G)
		degreeList, degreeList_ = exec_bfs(self.G,self.incidence_size,self.workers,self.calcUntilLayer)
		return degreeList, degreeList_

	def preprocess_neighbors_with_bfs_compact(self):
		
		'''
		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_bfs_compact,self.G,self.workers,self.calcUntilLayer)
			
			job.result()
		'''
		
		degreeList, degreeList_ = exec_bfs_compact(self.G,self.incidence_size,self.workers,self.calcUntilLayer)
		return degreeList, degreeList_

	def preprocess_degree_lists(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(preprocess_degreeLists)
			
			job.result()

		return


	def create_vectors(self):
		degrees = {}
		degrees_sorted = set()
		G = self.G
		for v in G.keys():
			degree = len(G[v])
			degrees_sorted.add(degree)
			if(degree not in degrees):
				degrees[degree] = {}
				degrees[degree]['vertices'] = deque() 
			degrees[degree]['vertices'].append(v)
		degrees_sorted = np.array(list(degrees_sorted),dtype='int')
		degrees_sorted = np.sort(degrees_sorted)

		l = len(degrees_sorted)
		for index, degree in enumerate(degrees_sorted):
			if(index > 0):
				degrees[degree]['before'] = degrees_sorted[index - 1]
			if(index < (l - 1)):
				degrees[degree]['after'] = degrees_sorted[index + 1]
		
		#saveVariableOnDisk(degrees,'degrees_vector')
		return degrees


	def calc_distances_all_vertices(self,degreeList,compactDegree = False):


		futures = {}

		count_calc = 0

		G = self.G
		vertices = G.keys()

		degrees = self.create_vectors()
		vertices_, degreeList = splitDegreeList(G,degreeList,degrees,compactDegree)
		#print('degree list')
		#print(degreeList)
		vertices = list(reversed(sorted(self.G.keys())))
		'''
		if(compactDegree):
		    degreeList = restoreVariableFromDisk('compactDegreeList')
		else:
		    degreeList = restoreVariableFromDisk('degreeList')
		'''

		parts = self.workers
		chunks = partition(vertices,parts)

		t0 = time()

		'''
		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				list_v = []
				for v in c:
					list_v.append([vd for vd in degreeList.keys() if vd > v])
				job = executor.submit(calc_distances_all, c, list_v, degreeList,part, compactDegree = compactDegree)
				futures[job] = part
				part += 1

			for job in as_completed(futures):
				job.result()
				r = futures[job]
		'''
		list_v = []
		for v in vertices:
			list_v.append([vd for vd in degreeList.keys() if vd > v])
		distances = calc_distances_all(vertices, list_v, degreeList,compactDegree = False)
		t1 = time()
		
		
		return vertices_, degreeList, distances

	def calc_distances_alternative(self, degreeList, degreeList_, compactDegree = False):

		futures = {}
		#distances = {}

		count_calc = 0

		G = self.G
		vertices = G.keys()

		degrees = self.create_vectors()

		vertices_, degreeList = splitDegreeList(G,degreeList,degrees,compactDegree)
		
		distances = calc_distances_alternative(vertices_, degreeList_)

		return vertices_, degreeList, distances

	def calc_distances(self, degreeList, compactDegree = False):

		futures = {}
		#distances = {}

		count_calc = 0

		G = self.G
		vertices = G.keys()
		
		'''
		parts = self.workers
		chunks = partition(vertices,parts)
		
		with ProcessPoolExecutor(max_workers = 1) as executor:

			part = 1
			for c in chunks:
				job = executor.submit(splitDegreeList,part,c,G,compactDegree)
				job.result()
				part += 1

		
		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				job = executor.submit(calc_distances, part, compactDegree = compactDegree)
				futures[job] = part
				part += 1

			for job in as_completed(futures):
				job.result()
				r = futures[job]
		'''

		degrees = self.create_vectors()

		vertices_, degreeList = splitDegreeList(G,degreeList,degrees,compactDegree)

		distances = calc_distances(vertices_, degreeList)

		return vertices_, degreeList, distances

	def consolide_distances(self):

		distances = {}

		parts = self.workers
		for part in range(1,parts + 1):
			d = restoreVariableFromDisk('distances-'+str(part))
			preprocess_consolides_distances(distances)
			distances.update(d)


		preprocess_consolides_distances(distances)
		saveVariableOnDisk(distances,'distances')
		return distances


	def create_distances_network(self, vertices_, degreeList, distances):
		
		'''
		with ProcessPoolExecutor(max_workers=1) as executor:
			job = executor.submit(generate_distances_network,self.workers)

			job.result()
		'''
		
		weights, alias_method_j, alias_method_q, graph_c, alias_method_j_c, alias_method_q_c = generate_distances_network(vertices_, degreeList, distances)

		return weights, alias_method_j, alias_method_q, graph_c, alias_method_j_c, alias_method_q_c

	def preprocess_parameters_random_walk(self, weights, alias_method_j, alias_method_q):
		'''
		with ProcessPoolExecutor(max_workers=1) as executor:
			job = executor.submit(generate_parameters_random_walk,self.workers)

			job.result()
		'''

		amount_neighbors = generate_parameters_random_walk(weights, self.workers)
		return amount_neighbors


	def simulate_walks(self,name, graph_c,alias_method_j, alias_method_q,amount_neighbors,num_walks,walk_length):

		# for large graphs, it is serially executed, because of memory use.
		if(len(self.G) > 500000):
			'''
			with ProcessPoolExecutor(max_workers=1) as executor:
				job = executor.submit(generate_random_walks_large_graphs,num_walks,walk_length,self.workers,self.G.keys())
				#job = executor.submit(generate_random_walks_large_graphs,num_walks,walk_length,self.workers, list(self.G.keys()))

				job.result()
			'''
			generate_random_walks_large_graphs(name, graph_c, alias_method_j, alias_method_q, amount_neighbors,num_walks,walk_length,self.workers,self.G.keys())

		else:
			'''
			with ProcessPoolExecutor(max_workers=1) as executor:
				#job = executor.submit(generate_random_walks,num_walks,walk_length,self.workers,self.G.keys())
				job = executor.submit(generate_random_walks,num_walks,walk_length,self.workers, list(self.G.keys()))
				#print(self.G.keys())
				job.result()
			'''
			generate_random_walks(name, graph_c, alias_method_j, alias_method_q, amount_neighbors,num_walks,walk_length,self.workers, list(self.G.keys()))

		return	





		

      	


