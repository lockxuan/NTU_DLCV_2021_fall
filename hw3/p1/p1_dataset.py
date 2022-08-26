import torch
from torch.utils.data import Dataset, DataLoader
import os, glob
from PIL import Image
import torchvision.transforms as T







class p1_dataset(Dataset):
	"""docstring for p1_dataset"""
	def __init__(self, data_path, train=False, transform=None):
		super(p1_dataset, self).__init__()

		self.imgs_path = glob.glob(os.path.join(data_path, '*.jpg'))
		self.train = train
		self.transform = transform

	def __getitem__(self, index):
		path = self.imgs_path[index]

		img = Image.open(path).convert('RGB')

		if self.transform:
			img = self.transform(img)
		else:
			img = T.Resize((224, 224))(img)
			img = T.ToTensor()(img)

		if self.train:
			return img, int(path.split('/')[-1].split('_')[0])

		return img, -1

	def __len__(self):
		return len(self.imgs_path)
		






if __name__ == '__main__':
	dataset = p1_dataset('../hw3_data/p1_data/train', train=True)

	p1_loader = DataLoader(dataset, batch_size=8, shuffle=True)
	for batch in p1_loader:
		imgs, labels = batch
		print(labels)
		break


