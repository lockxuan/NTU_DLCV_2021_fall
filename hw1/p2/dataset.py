import os, glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T



def mask_class(img):
	img = T.ToTensor()(img)
	img = 4 * img[0] + 2 * img[1] + img[2]
	mask = torch.zeros(img.shape, dtype=torch.long)
	mask[img == 3] = 0  # (Cyan: 011) Urban land 
	mask[img == 6] = 1  # (Yellow: 110) Agriculture land 
	mask[img == 5] = 2  # (Purple: 101) Rangeland 
	mask[img == 2] = 3  # (Green: 010) Forest land 
	mask[img == 1] = 4  # (Blue: 001) Water 
	mask[img == 7] = 5  # (White: 111) Barren land 
	mask[img == 0] = 6  # (Black: 000) Unknown 
	
	return mask



class p2_dataset(Dataset):
	"""docstring for p2_dataset"""
	def __init__(self, path, train=False, transform=None):
		super(p2_dataset, self).__init__()

		self.sat_list = glob.glob(os.path.join(path, '*.jpg'))
		self.sat_list.sort()
		self.mask_list = [sat.replace('sat', 'mask', -1).replace('jpg', 'png') for sat in self.sat_list]

		self.train = train
		self.transform = transform


	def __getitem__(self, index):
		sat = self.sat_list[index]
		sat = Image.open(sat)

		if self.transform:
			sat = self.transform(sat)

		if self.train:
			mask = self.mask_list[index]
			mask = Image.open(mask)
			mask = mask_class(mask)
		else:
			mask = self.mask_list[index].split('/')[-1]

		return sat, mask


	def __len__(self):
		return len(self.sat_list)



def p2_dataloader(path, batch_size=16, train=False, shuffle=False, transform=None):
	dataset = p2_dataset(path, train=train, transform=transform)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

	return dataloader





if __name__ == '__main__':
	#p2_dataset = p2_dataset('./hw1_data/p2_data/train', train=True)
	transform = T.Compose([
		T.ToTensor()
		])
	p2_dataloader = p2_dataloader('./hw1_data/p2_data/train', train=True, shuffle=True, transform=transform)
	for batch in p2_dataloader:
		print(batch)
		break








		

