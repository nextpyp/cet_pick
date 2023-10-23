import numpy as np 
import collections

def preprocess_label_constraints(labels):
	"""
	input a d * w * h numpy label map and output must-link and cannot-links 

	"""
	# flatten it into dwh, 1
	labels_f = labels.flatten()
	# labels_f = np.expand_dims(labels, axis = 1)

	# get unique labels 

	unique_lb = np.unique(labels_f)

	#traverse_lbs to get all lb subgroups
	lb_subgroups = {}
	for l in unique_lb:
		if l != 0:
			lb_sub = np.where(labels_f == l)
			
			# lb_subgroups.append(lb_sub)
			lb_subgroups[l] = lb_sub[0]

	#get must-link and cannot-links 
	num_of_neighbors = len(unique_lb) - 1

	ml_graph, cl_graph = collections.defaultdict(set), collections.defaultdict(set)

	#must links 
	keys = np.array(list(lb_subgroups.keys()))
	for k in keys:
		non_nei = keys[keys!=k]
		#first add neighbors
		nei = lb_subgroups[k]
		
		for i in nei:
			
			for j in nei:
				if j != i:
					ml_graph[i].add(j)
			for n in non_nei:
				non_n_l = lb_subgroups[n]
				cl_graph[i].update(non_n_l)
	for k, v in ml_graph.items():
		ml_graph[k] = list(v)
	for k, v in cl_graph.items():
		cl_graph[k] = list(v)
	return ml_graph, cl_graph
		

	







