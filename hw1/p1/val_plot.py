import torch
import torch.nn as nn
import os, glob
import numpy as np
import torchvision.transforms as T
import argparse

from dataset import p1_dataloader
from model import p1_model

import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.cm as cm



def load_model(model, modelPath, device):
	model.load_state_dict(torch.load(modelPath, map_location=device))
	return model


def inference(model, test_loader, device, model_path):

	model = load_model(model, model_path, device)
	model.eval()

	test_acc = 0
	test_len = 0

	with torch.no_grad():
		for step, batch in enumerate(test_loader):
			imgs, labels = batch
			imgs = imgs.to(device)
			labels = labels.to(device)

			output = model(imgs)

			_, pred = torch.max(output, 1)

			test_acc += (pred==labels).sum().item()
			test_len += len(labels)


			print('testing progress: {}/{}'.format(step+1, len(test_loader)), end='\r')
			print()

		test_acc /= test_len
		print('val acc: {:2.2%}'.format(test_acc))


def plot_features(model, test_loader, device, model_path):
	model = load_model(model, model_path, device)
	model.eval()

	target_features = []
	all_labels = []
	def hook(module, fin, fout):
		fs = torch.flatten(fout, 1)
		target_features.append(fs.detach().cpu().numpy())

	handler = model.resnet.fc[0].register_forward_hook(hook)

	with torch.no_grad():
		for step, batch in enumerate(test_loader):
			imgs, labels = batch
			imgs = imgs.to(device)
			all_labels.append(labels.detach().cpu().numpy())

			output = model(imgs)

			print('testing progress: {}/{}'.format(step+1, len(test_loader)), end='\r')

	handler.remove()

	features = np.concatenate(target_features)
	all_labels = np.concatenate(all_labels)
	print("Shape of latent features of testing images:", features.shape)

	tsne = TSNE(n_components=2, perplexity=10)
	reduced = tsne.fit_transform(features)
	colors = cm.get_cmap('hsv', 256)
	colors = colors(np.linspace(0.05, 0.95, 50))


	plt.figure()
	for i in range(50):
		selected = reduced[np.where(all_labels == i)[0]]
		plt.scatter(selected[:, 0], selected[:, 1], color=colors[i])

	plt.savefig('p1_tsne.png')


def main(args):
	
	transform = T.Compose([
		T.Resize(224),
		T.ToTensor(),
		T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
	])

	
	test_loader = p1_dataloader(args.test_path, batch=32, train=True, shuffle=False, transform=transform)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = p1_model().to(device)

	#inference(model, test_loader, device, model_path='p1_model.pth')
	plot_features(model, test_loader, device, model_path='p1_model.pth')







if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--test_path", default='../hw1_data/p1_data/val_50')
	args = parser.parse_args()
	main(args)


