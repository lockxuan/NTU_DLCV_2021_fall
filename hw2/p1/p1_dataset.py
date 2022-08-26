import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import os, glob
from PIL import Image
import torchvision.transforms as T


class p1_dataset(Dataset):
	"""docstring for p1_dataset"""
	def __init__(self, path, transform=None):
		super(p1_dataset, self).__init__()
		
		self.files = glob.glob(os.path.join(path, '*.png'))
		self.transform = transform

	def __getitem__(self, index):
		img = Image.open(self.files[index])

		if self.transform:
			img = self.transform(img)
		else:
			img = T.ToTensor()(img)


		return img

	def __len__(self):
		return len(self.files)




def p1_dataloader(path, batch_size=32, transform=None):
	dataset = p1_dataset(path, transform)

	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return dataloader











if __name__ == '__main__':
	#dataset = p1_dataset('./hw2_data/face/train')
	dataloader = p1_dataloader('./hw2_data/face/train')
	print(len(dataloader))
	for batch in dataloader:
		#print(batch)
		break




