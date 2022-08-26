import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import os, glob
import pandas as pd
from PIL import Image
import torchvision.transforms as T


class p2_dataset(Dataset):
	"""docstring for p2_dataset"""
	def __init__(self, img_folder, csv_path, train=False, transform=None):
		super(p2_dataset, self).__init__()
		
		#self.files = glob.glob(os.path.join(img_path, '*.png'))
		self.img_folder =img_folder
		self.csv = pd.read_csv(csv_path)

		self.train = train
		
		self.transform = transform

	def __getitem__(self, index):
		img_path = os.path.join(self.img_folder, self.csv['image_name'][index])
		img = Image.open(img_path)


		if self.transform:
			img = self.transform(img)
		else:
			img = T.ToTensor()(img)

		if self.train:
			label = self.csv['label'][index]

			return img, label

		return img

	def __len__(self):
		return len(self.csv)




def p2_dataloader(img_folder, csv_path, train=False, batch_size=32, transform=None):
	dataset = p2_dataset(img_folder, csv_path, train, transform)

	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return dataloader


class eval_dataset(Dataset):
	"""docstring for eval_dataset"""
	def __init__(self, img_folder, transform=None):
		super(eval_dataset, self).__init__()

		self.img_files = glob.glob(os.path.join(img_folder, '*.png'))

		self.transform = transform

	def __getitem__(self, index):
		img = Image.open(self.img_files[index])


		if self.transform:
			img = self.transform(img)
		else:
			img = T.ToTensor()(img)

		label = int(self.img_files[index].split('/')[-1].split('_')[0])

		return img, label


	def __len__(self):
		return len(self.img_files)








if __name__ == '__main__':
	dataset = p2_dataset('./hw2_data/digits/mnistm/train', './hw2_data/digits/mnistm/train.csv', train=True)
	print(dataset[0])
	#dataloader = p2_dataloader('./hw2_data/digits/mnistm/train', './hw2_data/digits/mnistm/train.csv')
	#print(len(dataloader))
	#for batch in dataloader:
		#print(batch)
	#	break




