from p1_dataset import p1_dataset
from p1_model import p1_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import os

import random
import argparse

from RandAugment import RandAugment



def fix_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	


def train(args, train_transform, val_transform, device):
	train_dataset = p1_dataset(args.train_path, train=True, transform=train_transform)
	val_dataset = p1_dataset(args.val_path, train=True, transform=val_transform)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


	#model = ViT_cus(img_size=384,patch_size=16, emb_dim=512, num_layers=12, num_heads=4, dropout=0.1).to(device)
	model = p1_model().to(device)
	if args.pretrained !=  None:
		model.load_state_dict(torch.load(args.pretrained, map_location=device))
	criterion = nn.CrossEntropyLoss()
	if args.optimizer == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.999))
	elif args.optimizer == 'adamw':
		optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
	elif args.optimizer == 'sgd':
		optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.8)


	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.7, milestones = [2,4,6,10])

	threshold = args.threshold

	for epoch in range(args.epochs):
		scheduler.step()
		train_loss = 0
		train_correct = 0
		train_total = 0
		val_loss = 0
		val_correct = 0
		val_total = 0

		### train epoch
		model.train()
		for step, batch in enumerate(train_loader):
			imgs, labels = batch
			imgs, labels = imgs.to(device), labels.to(device)

			optimizer.zero_grad()

			output = model(imgs)
			loss = criterion(output, labels)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			train_loss += loss.item()

			preds = torch.max(output, dim=1)[1]
			correct = (preds==labels).sum()
			train_correct += correct
			train_total += len(labels)

			print('training... step: {}/{} | loss: {:.4f} | acc: {:.2%}'.format(
				step+1, len(train_loader), loss, correct/len(labels)), end='\r')


		### validate epoch
		model.eval()
		with torch.no_grad():
			for step, batch in enumerate(val_loader):
				imgs, labels = batch
				imgs, labels = imgs.to(device), labels.to(device)

				output = model(imgs)
				loss = criterion(output, labels)

				val_loss += loss

				preds = torch.max(output, dim=1)[1]
				correct = (preds==labels).sum()
				val_correct += correct
				val_total += len(labels)

				print('validating... step: {}/{} | loss: {:.4f} | acc: {:.2%}'.format(
					step+1, len(val_loader), loss, correct/len(labels)), end='\r')


		print('epoch: {}/{} | train loss: {:.4f} | val loss: {:.4f} | train acc: {:.2%} | val acc: {:.2%}'.format(
			epoch+1, args.epochs, train_loss/len(train_loader), val_loss/len(val_loader), 
			train_correct/train_total, val_correct/val_total))



		if val_correct/val_total > threshold:
			threshold = val_correct/val_total
			save_dir = os.path.dirname(args.model_path)
			if save_dir != '':
				os.makedirs(save_dir, exist_ok=True)

			torch.save(model.state_dict(), args.model_path)
			print('---------- save model at epoch: {} ----------'.format(epoch+1))













def main(args):
	fix_seed(args.seed)

	train_transform = T.Compose([
		#T.Resize((384,384)),
		#T.Scale(420),
		T.RandomResizedCrop(384, scale=(0.7, 1.0)),
		#T.ColorJitter(brightness=0.2, hue=0.3, contrast=0.2),
		T.ColorJitter(brightness=0.3),
		T.RandomHorizontalFlip(p=0.5),
		T.RandomRotation(15),
		#T.RandAugment(),
		T.ToTensor(),
		#T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])
	#train_transform.transforms.insert(0, RandAugment(2, 15))

	val_transform = T.Compose([
		#T.Resize((224,224)),
		T.Resize((384,384)),
		#T.Scale(420),
		#T.CenterCrop(384),
		T.ToTensor(),
		#T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])


	#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	device = torch.device(args.device)

	

	train(args, train_transform, val_transform, device)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_path', type=str, default='../hw3_data/p1_data/train')
	parser.add_argument('--val_path', type=str, default='../hw3_data/p1_data/val')
	parser.add_argument('--model_path', type=str, default='./ckpt/16_p1_model.pth')
	parser.add_argument('--pretrained', type=str, default=None)
	parser.add_argument('--threshold', type=float, default=0.94)
	parser.add_argument('--batch_size', type=int, default=6)
	parser.add_argument('--epochs', type=int, default=16)
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.003)
	parser.add_argument('--optimizer', type=str, default='adam')
	parser.add_argument('--seed', type=int, default=10)
	parser.add_argument('--device', type=str, default='cuda:0')

	args = parser.parse_args()

	main(args)









