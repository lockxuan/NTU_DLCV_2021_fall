import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image





class p1_dataset(Dataset):
	"""docstring for p1_dataset"""
	def __init__(self, path, train=False, transform=None):
		super(p1_dataset, self).__init__()

		self.img_list = glob.glob(os.path.join(path, '*.png'))
		self.train = train
		self.transform = transform


	def __getitem__(self, index):
		file = self.img_list[index]

		img = Image.open(file)
		if self.transform:
			img = self.transform(img)

		if self.train:
			label = int(file.split('/')[-1].split('_')[0])
		else:
			label = file.split('/')[-1]

		return img, label


	def __len__(self):
		return len(self.img_list)




def p1_dataloader(path, batch=16, train=False, shuffle=False, transform=None):
	dataset = p1_dataset(path, train=train, transform=transform)
	dataloader = DataLoader(dataset, batch_size=batch, shuffle=shuffle)

	return dataloader

		






