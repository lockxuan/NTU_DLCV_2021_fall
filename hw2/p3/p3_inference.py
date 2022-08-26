import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision
import argparse
import random
import numpy as np

import os, glob

from p3_model import Extractor, Classifier
from PIL import Image
import csv



class p3_dataset(Dataset):
	"""docstring for p3_dataset"""
	def __init__(self, img_folder, transform=None):
		super(p3_dataset, self).__init__()
		
		self.img_files = glob.glob(os.path.join(img_folder, '*.png'))
		self.img_files.sort()
		
		self.transform = transform

	def __getitem__(self, index):
		img = Image.open(self.img_files[index])

		if self.transform:
			img = self.transform(img)
		else:
			img = T.ToTensor()(img)

		return img, self.img_files[index].split('/')[-1]

	def __len__(self):
		return len(self.img_files)


def fix_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def inference(extractor, classifier, target_loader, device, output_csv):

	extractor.eval()
	classifier.eval()

	if os.path.dirname(output_csv) != '':
		os.makedirs(os.path.dirname(output_csv), exist_ok=True)

	with open(output_csv, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['image_name', 'label'])
		for step, batch in enumerate(target_loader):
			imgs, file_names = batch
			imgs = imgs.to(device)


			features = extractor(imgs)
			outputs = classifier(features)

			preds = torch.max(outputs, dim=1)[1]

			for file_name, out in zip(file_names, preds.cpu().detach().numpy()):
					writer.writerow([file_name, out])

			print('inference step: {}/{}'.format(step+1, len(target_loader)), end='\r')
		






def main(args):
	fix_seed(args.seed)

	eval_transform = T.Compose([
		T.Resize(28),
		T.Grayscale(3),
		T.ToTensor(),
		T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
		])

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	extractor = Extractor().to(device)
	classifier = Classifier().to(device)


	if args.target_name =='mnistm':
		extractor_model_path = './ckpt/p3_s2m_e.pth'
		classifier_model_path = './ckpt/p3_s2m_c.pth'
	elif args.target_name =='svhn':
		extractor_model_path = './ckpt/p3_u2s_e.pth'
		classifier_model_path = './ckpt/p3_u2s_c.pth'
	elif args.target_name =='usps':
		extractor_model_path = './ckpt/p3_m2u_e.pth'
		classifier_model_path = './ckpt/p3_m2u_c.pth'

	#target_loader = p3_dataloader(args.target_path, batch_size=64, transform=eval_transform)

	dataset = p3_dataset(args.target_path, transform=eval_transform)
	target_loader = DataLoader(dataset, batch_size=64, shuffle=False)
	

	extractor.load_state_dict(torch.load(extractor_model_path, map_location=device))
	classifier.load_state_dict(torch.load(classifier_model_path, map_location=device))




	inference(extractor, classifier, target_loader, device, args.output_csv)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--target_path", default='./hw2_data/digits/usps/test')
	parser.add_argument("--target_name", type=str, choices=['mnistm','svhn','usps'], default='usps')
	parser.add_argument("--output_csv", default='./r09942091_p3/p3_s2m.csv')
	parser.add_argument("--seed", type=int, default=2022)

	args = parser.parse_args()
	main(args)



