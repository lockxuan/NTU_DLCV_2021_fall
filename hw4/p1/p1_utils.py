import numpy as np
import torch
import torch.nn as nn


def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)


def cal_distance(method='euclidean', model=None):
	def euc(a, b):
		n = a.size(0) # n_way*k_query
		m = b.size(0) # n_way*k_shot
		
		a = a.unsqueeze(1).expand(n, m, -1) # (n, 1, 64) -> (n, m, 64)
		b = b.unsqueeze(0).expand(n, m, -1) # (1, m, 64) -> (n, m, 64)

		distance = ((a-b)**2).sum(dim=2).sqrt() # n_way*k_query, n_way*k_shot

		return -distance

	def cosine(a, b):
		a = a.unsqueeze(1)
		b = b.unsqueeze(0)

		distance = nn.functional.cosine_similarity(a, b, dim=2)
		return -distance

	if method == 'euclidean' or method == 'euc':
		return euc
	elif method == 'cosine' or method == 'cos':
		return cosine
	elif method == 'parametric' or method == 'para':
		return model.cal_distance



def cal_accuracy(outputs, labels):
	preds = torch.max(outputs, dim=1)[1]

	correct = (preds==labels).sum()
	accuracy = correct/len(labels)

	return accuracy.item(), correct, len(labels)


	