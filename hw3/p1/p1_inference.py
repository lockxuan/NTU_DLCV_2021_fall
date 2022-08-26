from p1_model import p1_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
import os, glob
from PIL import Image

import random
import argparse

import csv



def fix_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


class inference_dataset(Dataset):
	"""docstring for inference_dataset"""
	def __init__(self, data_path, transform=None):
		super(inference_dataset, self).__init__()

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

		return img, path.split('/')[-1]

	def __len__(self):
		return len(self.imgs_path)
	


def train(args, val_transform, device):
	val_dataset = inference_dataset(args.val_path, transform=val_transform)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


	model = p1_model().to(device)
	if args.model_path !=  None:
		model.load_state_dict(torch.load(args.model_path, map_location=device))

	if os.path.dirname(args.output_csv) != '':
		os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)


	### validate epoch
	model.eval()
	with torch.no_grad():
		with open(args.output_csv, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['image_id', 'label'])
			for step, batch in enumerate(val_loader):
				imgs, file_names = batch
				imgs = imgs.to(device)

				output = model(imgs)

				preds = torch.max(output, dim=1)[1]

				for file_name, out in zip(file_names, preds.cpu().detach().numpy()):
					writer.writerow([file_name, out])

				print('inference step: {}/{}'.format(step+1, len(val_loader)), end='\r')






def main(args):
	fix_seed(args.seed)

	val_transform = T.Compose([
		T.Resize((384,384)),
		T.ToTensor(),
		T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])


	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	train(args, val_transform, device)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--val_path', type=str, default='../hw3_data/p1_data/val')
	parser.add_argument('--model_path', type=str, default='./ckpt/p1_model.pth')
	parser.add_argument("--output_csv", default='./r09942091_p1_inference.csv')
	parser.add_argument('--batch_size', type=int, default=6)
	parser.add_argument('--seed', type=int, default=10)
	parser.add_argument('--device', type=str, default='cuda:0')

	args = parser.parse_args()

	main(args)









