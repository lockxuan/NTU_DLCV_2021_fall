import torch
import torch.nn as nn
import os, glob
import cv2
import numpy as np
import torchvision.transforms as T
import argparse

from dataset import p1_dataloader
from model import p1_model



def save_model(model, save):
	torch.save(model.state_dict(), save)

def train(epochs, model, train_loader, test_loader, optimizer, lr_scheduler, criterion, device, save=None):

	for epoch in range(epochs):

		train_loss = 0
		train_acc = 0
		train_len = 0
		train_output = []

		test_loss = 0
		test_acc = 0
		test_len = 0
		test_output = []

		global_acc = 0.7

		model.train()
		for step, batch in enumerate(train_loader):
			imgs, labels = batch
			imgs = imgs.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()

			output = model(imgs)
			loss = criterion(output, labels)
			train_loss += loss.item()

			loss.backward()
			optimizer.step()

			_, pred = torch.max(output, 1)
			train_acc += (pred==labels).sum().item()
			train_len += len(labels)


			print('training progress: {}/{}'.format(step+1, len(train_loader)), end='\r')

		model.eval()
		with torch.no_grad():
			for step, batch in enumerate(test_loader):
				imgs, labels = batch
				imgs = imgs.to(device)
				labels = labels.to(device)

				output = model(imgs)
				loss = criterion(output, labels)
				test_loss += loss.item()

				_, pred = torch.max(output, 1)
				test_acc += (pred==labels).sum().item()
				test_len += len(labels)

				print('testing progress: {}/{}'.format(step+1, len(test_loader)), end='\r')

		lr_scheduler.step()

		train_loss /= len(train_loader)
		test_loss /= len(test_loader)
		train_acc /= train_len
		test_acc /= test_len


		print('epoch: {} | train loss: {:3f} | train acc: {:2.2%} | val loss: {:3f} | val acc: {:2.2%}'
			.format(epoch+1, train_loss, train_acc, test_loss, test_acc))

		if test_acc>global_acc and save:
			global_acc = test_acc
			save_model(model, save)




def main(args):
	
	transform = {'train': T.Compose([
		T.Resize(224),
		T.RandomRotation(degrees=15, resample=Image.BILINEAR),
		T.RandomHorizontalFlip(),
		T.ToTensor(),
		T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
	]), 'val': T.Compose([
		T.Resize(224),
		T.ToTensor(),
		T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
	])
	}

	
	train_loader = p1_dataloader(args.train_path, batch=32, train=True, shuffle=True, transform=transform['train'])
	test_loader = p1_dataloader(args.test_path, batch=32, train=True, shuffle=False, transform=transform['val'])


	device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
	epochs = 50
	lr = 0.001

	model = p1_model().to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.6)

	train(epochs, model, train_loader, test_loader, optimizer, lr_scheduler, criterion, device, save='p1_model.pth')









if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_path", default='../hw1_data/p1_data/train_50')
	parser.add_argument("--test_path", default='../hw1_data/p1_data/val_50')
	args = parser.parse_args()
	main(args)


