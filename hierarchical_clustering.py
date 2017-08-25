#!/usr/bin/python
'''
Hierarchical clustering implementation for both Agglomerative (bottom-up) and Devisive (top-down)
Different types of clustere distance measures
(i):-   Single Link (min distance)
(ii):-  Complete Link (max distance)
(iii):- Average Link (avg distance)

Author ;- Basant Kumar
'''

import numpy as np 
import sys
import math
import os
import heapq
import itertools



class Hierarchical_Clustering:
	def __init__(self, c_type="a", dist_measure="min", input_file_name="iris-data", k=1):
		self.c_type=c_type
		self.dist_measure=dist_measure
		self.input_file_name=input_file_name
		self.k=k
		self.dataset=None
		self.dataset_size=0
		self.dimension=0
		self.heap=[]
		self.clusters=[]
		self.gold_standard=0

	def initialize(self):
		#Initialize and check parameters
		if not os.path.isfile(self.input_file_name):
			self.quit("input file doesn't exists or it's not file")

		self.dataset, self.clusters, self.gold_standard = self.load_data(self.input_file_name)
		self.dataset_size = len(self.dataset)

		if self.dataset_size == 0:
		    self.quit("Input file doesn't include any data")

		if self.k == 0:
		    self.quit("k = 0, no cluster will be generated")

		if self.k > self.dataset_size:
		    self.quit("k is larger than the number of existing clusters")

		self.dimension = len(self.dataset[0]["data"])

		if self.dimension == 0:
		    self.quit("dimension for dataset cannot be zero")

	def euclidean_distance(self, data_point_one, data_point_two):
		size=len(data_point_one)
		result=0.0
		for i in range(size):
			f1=float(data_point_one[i])
			f2=float(data_point_two[i])
			tmp=f1-f2
			result+=pow(tmp,2)
		result=math.sqrt(result)
		return result



	def compute_pairwise_distance(self, dataset):
		result=[]
		dataset_size=len(dataset)

		for i in range(dataset_size-1):
			for j in range(i+1,dataset_size):
				dist=self.euclidean_distance(dataset[i]["data"],dataset[j]["data"])
				result.append((dist,[dist,[[i],[j]]]))
		#print(result)
		return result

	def compute_centroid(self, dataset, data_points_index):
		size=len(data_points_index)
		dim=self.dimension
		centroid=[0.0]*dim

		for idx in data_points_index:
			dim_data=dataset[idx]["data"]
			for i in range(dim):
				centroid[i]+=float(dim_data[i])
		for i in range(dim):
			centroid[i]/=size
		return centroid	

	def build_priority_queue(self, distance_list):
		heapq.heapify(distance_list)
		self.heap=distance_list
		return self.heap

	def valid_heap_node(self, heap_node, old_clusters):
		pair_dist=heap_node[0]
		pair_data=heap_node[1]

		for old_cluster in old_clusters:
			if old_cluster in pair_data:
				return False
		return True

	def add_heap_entry(self, heap, new_cluster, current_clusters):
		for ex_cluster in current_clusters.values():
			new_heap_entry=[]
			dist=self.euclidean_distance(ex_cluster["centroid"], new_cluster["centroid"])
			new_heap_entry.append(dist)
			new_heap_entry.append([new_cluster["elements"],ex_cluster["elements"]])
			heapq.heappush(heap, (dist, new_heap_entry))



	def hierarchical_clustering(self):
		dataset=self.dataset
		current_clusters=self.clusters
		old_clusters=[]
		heap=self.compute_pairwise_distance(dataset)
		heap=self.build_priority_queue(heap)

		while len(current_clusters) > self.k:
			dist, min_item = heapq.heappop(heap)
			pair_data = min_item[1]

			if not self.valid_heap_node(min_item, old_clusters):
				continue

			new_cluster={}
			new_cluster_elements=sum(pair_data, [])
			new_cluster_centroid= self.compute_centroid(dataset,new_cluster_elements)
			new_cluster_elements.sort()
			new_cluster.setdefault("centroid", new_cluster_centroid)
			new_cluster.setdefault("elements", new_cluster_elements)

			for pair_item in pair_data:
				old_clusters.append(pair_item)
				del current_clusters[str(pair_item)]
			self.add_heap_entry(heap, new_cluster, current_clusters)
			current_clusters[str(new_cluster_elements)]=new_cluster
		#current_clusters.sort()
		return current_clusters



	def load_data(self, input_file_name):
		"""
		load data and do some preparations

		"""
		input_file = open(input_file_name, 'rU')
		dataset = []
		clusters = {}
		gold_standard = {}
		id = 0
		for line in input_file:
			line = line.strip('\n')
			row = str(line)
			row = row.split(",")
			iris_class = row[-1]

			data = {}
			data.setdefault("id", id)   # duplicate
			data.setdefault("data", row[:-1])
			data.setdefault("class", row[-1])
			dataset.append(data)

			clusters_key = str([id])
			clusters.setdefault(clusters_key, {})
			clusters[clusters_key].setdefault("centroid", row[:-1])
			clusters[clusters_key].setdefault("elements", [id])

			gold_standard.setdefault(iris_class, [])
			gold_standard[iris_class].append(id)

			id += 1
		#print(np.array(dataset))
		#print(np.array(clusters))
		#print(np.array(gold_standard))
		return dataset, clusters, gold_standard

	def evaluate(self, current_clusters):
	    gold_standard = self.gold_standard
	    current_clustes_pairs = []

	    for (current_cluster_key, current_cluster_value) in current_clusters.items():
	        tmp = list(itertools.combinations(current_cluster_value["elements"], 2))
	        current_clustes_pairs.extend(tmp)
	    tp_fp = len(current_clustes_pairs)

	    gold_standard_pairs = []
	    for (gold_standard_key, gold_standard_value) in gold_standard.items():
	        tmp = list(itertools.combinations(gold_standard_value, 2))
	        gold_standard_pairs.extend(tmp)
	    tp_fn = len(gold_standard_pairs)

	    tp = 0.0
	    for ccp in current_clustes_pairs:
	        if ccp in gold_standard_pairs:
	            tp += 1

	    if tp_fp == 0:
	        precision = 0.0
	    else:
	        precision = tp/tp_fp
	    if tp_fn == 0:
	        precision = 0.0
	    else:
	        recall = tp/tp_fn

	    return precision, recall

	def quit(self, err_desc):
		raise SystemExit('\n'+ "PROGRAM EXIT: " + err_desc + ', please check your input' + '\n')	

	def display(self, current_clusters, precision, recall):
	    print precision
	    print recall
	    clusters = current_clusters.values()
	    for cluster in clusters:
	        cluster["elements"].sort()
	        print cluster["elements"]


def main():
	'''python HC.py -a/-c -min/-max/-avg input_file_name'''
	#x,y=data_preparation();
	c_type = sys.argv[1]
	dm = sys.argv[2]
	filename = sys.argv[3]     
	k = int(sys.argv[4])
	

	hc = Hierarchical_Clustering(c_type, dm, filename, k)
	hc.initialize()
	current_clusters=hc.hierarchical_clustering()
	precision, recall = hc.evaluate(current_clusters)
	hc.display(current_clusters, precision, recall)

	## euclidean_distance() test
	#loaded_data = hc.loaded_dataset()
	#print loaded_data
	#print hc.euclidean_distance(loaded_data[0]["data"],loaded_data[1]["data"])

	## compute_centroid() test
	#loaded_data = hc.loaded_dataset()
	#hc.compute_centroid(loaded_data, [10, 11, 12, 13])

	## distance_list test
	#distance_list = hc.compute_pairwise_distance()
	#distance_list.sort()
	#print distance_list

	## heapq test
	#heap = []
	#data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
	#data = [[1,4,5], [3,6,1], [5,6,10], [7,2,11], [9,6,1], [2,1,5], [4,2,1], [6,6,5], [8,7,1], [0,1,0]]
	#heapq.heapify(data)
	#print data



if __name__=="__main__":
	main()
