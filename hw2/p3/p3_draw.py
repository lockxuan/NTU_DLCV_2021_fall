import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision
import argparse
import random
import numpy as np

import os, glob

from p3_model import Extractor, Classifier
from p3_dataset import p3_dataloader
from PIL import Image
import csv

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.cm as cm



def fix_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def inference(extractor, classifier, source_loader, target_loader, device, output_path, source_target):

	extractor.eval()
	classifier.eval()

	source_latents = []
	source_labels = []

	target_latents = []
	target_labels = []

	with torch.no_grad():
		for step, batch in enumerate(source_loader):
			imgs, labels = batch
			imgs = imgs.to(device)


			features = extractor(imgs)
			features = classifier(features)
			source_latents.append(features.cpu())

			source_labels.append(labels)


			print('inference source step: {}/{}         '.format(step+1, len(source_loader)), end='\r')

		for step, batch in enumerate(target_loader):
			imgs, labels = batch
			imgs = imgs.to(device)


			features = extractor(imgs)
			features = classifier(features)
			target_latents.append(features.cpu())

			target_labels.append(labels)


			print('inference target step: {}/{}         '.format(step+1, len(target_loader)), end='\r')

		source_latents = torch.cat(source_latents, dim=0)
		source_labels = torch.cat(source_labels, dim=0)

		target_latents = torch.cat(target_latents, dim=0)
		target_labels = torch.cat(target_labels, dim=0)

		#min_len = min(len(source_labels), len(target_labels))
		min_len = 2000

		all_latents = torch.cat((source_latents[:min_len], target_latents[:min_len]), dim=0)
		all_labels = torch.cat((source_labels[:min_len], target_labels[:min_len]), dim=0)
		domain_labels = torch.cat((torch.ones(len(source_labels[:min_len])), torch.zeros(len(target_labels[:min_len]))), dim=0)

		tsne = TSNE(n_components=2)
		reduced_class = tsne.fit_transform(all_latents)

		os.makedirs(output_path, exist_ok=True)

		plt.figure()
		for i in range(10):
			selected = reduced_class[all_labels == i]
			plt.scatter(selected[:, 0], selected[:, 1], label=str(i))
		plt.legend()
		plt.savefig(output_path+'/'+source_target+'_class.png')

		plt.figure()
		for i in range(2):
			selected = reduced_class[domain_labels == i]
			plt.scatter(selected[:, 0], selected[:, 1], alpha=0.2, label=str(i))
		plt.legend()
		plt.savefig(output_path+'/'+source_target+'_label.png')
		

		






def main(args):
	fix_seed(args.seed)

	eval_transform = T.Compose([
		T.Resize(28),
		T.Grayscale(3),
		T.ToTensor(),
		T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
		])

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


	if args.source_target =='s2m':
		extractor_model_path = './p3/p3_s2m_e.pth'
		classifier_model_path = './p3/p3_s2m_c.pth'
	elif args.source_target =='u2s':
		extractor_model_path = './p3/p3_u2s_e.pth'
		classifier_model_path = './p3/p3_u2s_c.pth'
	elif args.source_target =='m2u':
		extractor_model_path = './p3/p3_m2u_e.pth'
		classifier_model_path = './p3/p3_m2u_c.pth'

	source_loader = p3_dataloader(args.source_path, shuffle=True, batch_size=64, transform=eval_transform)
	target_loader = p3_dataloader(args.target_path, shuffle=True, batch_size=64, transform=eval_transform)
	
	extractor = Extractor().to(device)
	classifier = Classifier().to(device)

	extractor.load_state_dict(torch.load(extractor_model_path, map_location=device))
	classifier.load_state_dict(torch.load(classifier_model_path, map_location=device))

	inference(extractor, classifier, source_loader, target_loader, device, args.output_path, args.source_target)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path", default='./hw2_data/digits/mnistm/test')
	parser.add_argument("--target_path", default='./hw2_data/digits/usps/test')
	parser.add_argument("--source_target", type=str, choices=['m2u','s2m','u2s'], default='m2u')
	parser.add_argument("--output_path", default='./p3/p3_plot')
	parser.add_argument("--seed", type=int, default=2022)

	args = parser.parse_args()
	main(args)



