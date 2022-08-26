import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import os, glob
import pandas as pd
from PIL import Image
import torchvision.transforms as T


class p3_dataset(Dataset):
	"""docstring for p3_dataset"""
	def __init__(self, img_folder, transform=None):
		super(p3_dataset, self).__init__()
		
		self.img_folder = img_folder
		self.csv = pd.read_csv(img_folder+'.csv')
		
		self.transform = transform

	def __getitem__(self, index):
		img_path = os.path.join(self.img_folder, self.csv['image_name'][index])
		img = Image.open(img_path)


		if self.transform:
			img = self.transform(img)
		else:
			img = T.ToTensor()(img)

		label = self.csv['label'][index]

		return img, label

	def __len__(self):
		return len(self.csv)




def p3_dataloader(img_folder, shuffle=False, batch_size=32, transform=None):
	dataset = p3_dataset(img_folder, transform)

	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
	return dataloader







if __name__ == '__main__':
	mnist = p3_dataset('./hw2_data/digits/mnistm/train', './hw2_data/digits/mnistm/train.csv', train=True)
	svhn = p3_dataset('./hw2_data/digits/svhn/train', './hw2_data/digits/svhn/train.csv', train=True)
	usps = p3_dataset('./hw2_data/digits/usps/train', './hw2_data/digits/usps/train.csv', train=True)
	print(mnist[0][1])
	print(svhn[0][1])
	print(usps[0][1])
	#dataloader = p2_dataloader('./hw2_data/digits/mnistm/train', './hw2_data/digits/mnistm/train.csv')
	#print(len(dataloader))
	#for batch in dataloader:
		#print(batch)
	#	break




