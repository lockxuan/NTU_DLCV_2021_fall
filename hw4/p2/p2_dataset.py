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
	T.Resize((128, 128)),
	T.RandomHorizontalFlip(p=20),
	T.ToTensor(),
	#T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

val_transform = T.Compose([
	T.Resize((128, 128)),
	T.ToTensor(),
	#T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])



class p2_dataset(Dataset):
	"""docstring for p2_dataset"""
	def __init__(self, path, transform):
		super(p2_dataset, self).__init__()
		self.path = path.rstrip('/')

		df = pd.read_csv(path+'.csv')
		df['gt'] = df['label'].astype('category').cat.codes ### generate gt from label

		self.imgs_path = [os.path.join(path, file_name) for file_name in df['filename']]
		self.labels = df['gt'].tolist()
		#print(self.labels)
		gt_to_label = {}
		for l, g in zip(df['label'].tolist(), df['gt'].tolist()):
			if g in gt_to_label:
				continue
			gt_to_label[g] = l
		print('asd')
		print(gt_to_label)


		self.transform = transform

	def __getitem__(self, index):
		label = self.labels[index]


		img = self.transform(Image.open(self.imgs_path[index]))

		return img, label

	def __len__(self):
		return len(self.imgs_path)



		
		






if __name__ == '__main__':
	import matplotlib.pyplot as plt
	torch.manual_seed(10)
	#random.seed(seed)
	np.random.seed(10)

	path = '../hw4_data/office/val'
	path2 = '../hw4_data/office/train'

	val_dataset = p2_dataset(path, val_transform)
	train_dataset = p2_dataset(path2, val_transform)

	#print(val_dataset[0])
	
	"""
	loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
	for j in range(1):
		for i, batch in enumerate(loader):
			imgs, labels = batch
			print(imgs.shape)
			print(labels)
			if i==3:
				break
			break

		break
	"""


