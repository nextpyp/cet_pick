import numpy as np 
import torch 
import torch.nn.functional as F
import random 
import faiss

class FaissKMeans(object):
	"""
	perform unconstrained k-means 

	"""

	def __init__(self, opt, n_clusters = 4, n_init = 10, max_iter = 100):
		self.n_clusters = n_clusters
		self.n_init = n_init
		self.max_iter = max_iter 
		self.kmeans = None 
		self.cluster_centers = None 
		self.inertia = None 

	def run_kmeans(self, embeddings):
		self.kmeans = faiss.Kmeans(d = embeddings.shape[1], k = self.n_clusters, niter = self.max_iter, nredo = self.n_init, spherical=True)
		self.kmeans.train(embeddings.astype(np.float32))

		self.cluster_centers = self.kmeans.centroids 
		self.inertia = self.kmeans.obj[-1]

		labels = self.kmeans.index.search(embeddings, 1)[1]

		return self.cluster_centers, labels

class MPKMeans(object):
	"""
	perform constrained k-means algorithms with must-links and no-links 

	"""

	def __init__(self, opt, n_clusters = 4, exist_labels = 2, max_iter = 2, w = 1, emb_dim = 16):
		super(MPKMeans, self).__init__()
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.w = w 
		self.emb_dim = emb_dim
		self.exist_labels = exist_labels
		self.opt = opt

	def _initialize_cluster_centers(self, embeddings, labels):
		# reshape embedding dim of 1, c, d, w, h into d*w*h, c
		# embeddings = embeddings.reshape((1, self.emb_dim, -1))
		# embeddings = embeddings.squeeze().T

		labels = labels.reshape(1, -1)
		labels = labels.squeeze().long()

		one_hot_labels = F.one_hot(labels).float()
	

		# lb * dim
		cluster_centers = torch.matmul(one_hot_labels.T, embeddings)

		cluster_centers_unlabeled = cluster_centers[0]
		cluster_centers_labeled = cluster_centers[1:]
		# exist_labels = 



		if self.n_clusters > self.exist_labels:
			additional_centroid_needed = self.n_clusters - self.exist_labels
			more_inits = torch.randn((additional_centroid_needed, self.emb_dim), device = self.opt.device) + cluster_centers_unlabeled
			tot_inits = torch.cat((cluster_centers_labeled, more_inits), 0)
			tot_inits = F.normalize(tot_inits, dim = 1)
		else:
			tot_inits = F.normalize(cluster_centers_labeled, dim=1)
		return tot_inits

	def _cosine_dist(self, u, v):
		# print('u', u.shape)
		# print('v',v.shape)
		u = F.normalize(u, dim=0)
		v = F.normalize(v, dim=0)
		sim = torch.matmul(u.T, v)
		# print('sim', sim)
		dist = 1 - sim
		return dist

	def _objective_fn(self, embeddings, i, labels, cluster_centers, cluster_id, ml_graph, cl_graph):
		# print('cluster_center', cluster_centers.shape)
		term_d = self._cosine_dist(embeddings[i], cluster_centers[cluster_id])
		# print('term_d', term_d)
		# print('embeddings shape', embeddings[i].shape)
		# print('cluster_centers shape', cluster_centers[cluster_id].shape)
		def f_m(i, j):
			return self._cosine_dist(i, j)

		def f_c(i, j):
			return 1 - self._cosine_dist(i, j)

		term_m = 0
		if i in ml_graph:
			for j in ml_graph[i]:
				if labels[j] >= 0 and labels[j] != cluster_id:
					# print('i',i)
					# print('j',j)
					# print('j',j[0])
					# print('embeddings[j]', embeddings[j])
					# print(embeddings[j[0]].shape)
					term_m += self.w * f_m(embeddings[i], embeddings[j[0]])
		term_c = 0
		if i in cl_graph:
			for j in cl_graph[i]:
				if labels[j] == cluster_id:
					# print('i',i)
					# print('j',j.item())
					# print('j',j[0])
					# print('embeddings[j]', embeddings[j])
					# print(embeddings[j[0]].shape)
					term_c += self.w * f_c(embeddings[i], embeddings[j[0]])

		return term_d + term_m + term_c

	def _assign_clusters(self, embeddings, cluster_centers, ml_graph, cl_graph):
		# embeddings = embeddings.reshape((1, self.emb_dim, -1))
		# embeddings = embeddings.squeeze().T

		labels = torch.full((embeddings.shape[0],1), -1)
		index = list(range(embeddings.shape[0]))
		#shuffle index each iteration for greedy assignment
		np.random.shuffle(index)

		for i in index:
			labels[i] = np.argmin([self._objective_fn(embeddings, i, labels, cluster_centers, cluster_id, ml_graph, cl_graph) for cluster_id, cluster_center in enumerate(cluster_centers)])
		labels = torch.tensor(labels)
		return labels

	def _calculate_prototypes_from_labels(self, embeddings, labels):

		# embeddings = embeddings.reshape((1, self.emb_dim, -1))
		# embeddings = embeddings.squeeze().T

		labels = labels.squeeze().long()

		one_hot_labels = F.one_hot(labels).float().to(device = self.opt.device)
		cluster_centers = torch.matmul(one_hot_labels.T, embeddings)
		# print('cluster_center pre nom', cluster_centers)
		prototypes = F.normalize(cluster_centers, dim=1)
		# print('prototypes', prototypes)
		return prototypes


	def mpkmeans_with_initial_labels(self, embeddings, initial_labels, ml_graph, cl_graph):
		cluster_centers = self._initialize_cluster_centers(embeddings, initial_labels)

		for iteration in range(self.max_iter):
			prev_cluster_centers = cluster_centers.clone().detach()
			labels = self._assign_clusters(embeddings, cluster_centers, ml_graph, cl_graph)

			cluster_centers = self._calculate_prototypes_from_labels(embeddings, labels)
			# cluster_centers_shift = (prev_cluster_centers - cluster_centers)
			converged = torch.allclose(cluster_centers, prev_cluster_centers, atol = 1e-6)

			if converged:
				break 

		return cluster_centers, labels


	def mpkmeans_with_old_centers(self, embeddings, cluster_centers, ml_graph, cl_graph):
		
		for iteration in range(self.max_iter):
			prev_cluster_centers = cluster_centers.clone().detach()
			labels = self._assign_clusters(embeddings, cluster_centers, ml_graph, cl_graph)

			cluster_centers = self._calculate_prototypes_from_labels(embeddings, labels)
			# cluster_centers_shift = (prev_cluster_centers - cluster_centers)
			converged = torch.allclose(cluster_centers, prev_cluster_centers, atol = 1e-6)

			if converged:
				break 

		return cluster_centers, labels





















