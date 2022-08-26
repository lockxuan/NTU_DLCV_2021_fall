import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

import torchvision.transforms as T
import pandas as pd
import os
from PIL import Image
import numpy as np




train_transform = T.Compose([
	T.Resize((84, 84)),
	T.RandomHorizontalFlip(p=20),
	T.ToTensor(),
	#T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

val_transform = T.Compose([
	T.Resize((84, 84)),
	T.ToTensor(),
	#T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])



class p1_dataset(Dataset):
	"""docstring for p1_dataset"""
	def __init__(self, path, transform):
		super(p1_dataset, self).__init__()
		self.path = path.rstrip('/')

		df = pd.read_csv(path+'.csv')
		df['gt'] = df['label'].astype('category').cat.codes ### generate gt from label

		self.imgs_path = [os.path.join(path, file_name) for file_name in df['filename']]
		self.labels = df['gt'].tolist()
		#print(self.labels)

		self.transform = transform

	def __getitem__(self, index):
		label = self.labels[index]


		img = self.transform(Image.open(self.imgs_path[index]))

		return img, label

	def __len__(self):
		return len(self.imgs_path)



class p1_sampler(Sampler):
	"""docstring for p1_sampler"""
	def __init__(self, labels, n_way=5, k_shot=1, k_query=10):
		self.n_way = n_way
		self.k_shot = k_shot
		self.k_query = k_query

		self.classes = set(labels)

		self.classes_idxs = []
		#self.total_batches = len(labels)//(self.n_way*self.n_samples)
		self.total_batches = 300
		
		labels = np.array(labels)
		for c in self.classes:
			idxs = np.argwhere(labels==c).reshape(-1)
			self.classes_idxs.append(torch.from_numpy(idxs))


	def __iter__(self):
		for _ in range(self.total_batches):
			shot_batch = torch.LongTensor(self.n_way * self.k_shot)
			query_batch = torch.LongTensor(self.n_way * self.k_query)
			n_way_classes = torch.randperm(len(self.classes))[:self.n_way]

			for i, c in enumerate(n_way_classes):
				s = slice(i*self.k_shot, (i+1)*self.k_shot)
				q = slice(i*self.k_query, (i+1)*self.k_query)

				target = self.classes_idxs[c]
				rand_idxs = torch.randperm(len(target))
				shots_idxs = rand_idxs[:self.k_shot]
				query_idxs = rand_idxs[self.k_shot:self.k_shot+self.k_query]

				shot_batch[s] = target[shots_idxs]
				query_batch[q] = target[query_idxs]

			#query_batch = query_batch[torch.randperm(len(query_batch))]
			batch = torch.cat((shot_batch, query_batch), dim=0)

			yield batch

		#indices = torch.cat(indices, dim=0)
		#return iter(indices)


	def __len__(self):
		return self.total_batches
		
		






if __name__ == '__main__':
	import matplotlib.pyplot as plt
	torch.manual_seed(10)
	#random.seed(seed)
	np.random.seed(10)

	path = '../hw4_data/mini/val'

	val_dataset = p1_dataset(path, val_transform)
	print(len(val_dataset))


	loader = DataLoader(val_dataset, num_workers=5, batch_sampler=p1_sampler(val_dataset.labels))
	print(len(loader))
	for j in range(1):
		iter_loader = iter(loader)
		for i, batch in enumerate(iter_loader):
			#print(i)
			#print(batch[1])
			imgs, labels = batch
			print(len(imgs))
			print(labels)
			#fig, axs = plt.subplots(len(imgs))
			for i in range(len(imgs)):
				#axs[i].imshow(imgs[i].permute(1,2,0))
				plt.subplot(2, 5, i+1)
				plt.imshow(imgs[i].permute(1,2,0))
			plt.show()
			break
			if i==2:
				break
		break



