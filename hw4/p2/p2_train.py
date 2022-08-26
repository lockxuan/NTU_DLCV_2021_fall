import torch
import torch.nn as nn
from torchvision import models
from p2_dataset import p2_dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
import argparse
import os

from p2_model import p2_model



def fix_seed(seed):
	pass


def train(args, feature_extractor, transform, device):
	train_dataset = p2_dataset(args.train_path, transform['train'])
	val_dataset = p2_dataset(args.val_path, transform['val'])
	train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

	if args.fix_backbone:
		model = p2_model().to(device)
	else:
		model = p2_model(feature_extractor).to(device)

	criterion = nn.CrossEntropyLoss()

	if args.fix_backbone:
		print('------- fix backbone -------')
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	else:
		optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + list(model.parameters()), lr=args.lr)

	threshold = args.threshold


	for epoch in range(args.epochs):
		train_loss = 0
		train_correct = 0
		train_total = 0

		model.train()
		if not args.fix_backbone:
			feature_extractor.train()
		for step, batch in enumerate(train_loader):
			imgs, labels = batch
			imgs, labels = imgs.to(device), labels.to(device)

			features = feature_extractor(imgs)
			outputs = model(features)

			loss = criterion(outputs, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss += loss.item()

			preds = torch.max(outputs, dim=1)[1]
			correct = (preds==labels).sum().item()
			train_correct += correct
			train_total += len(labels)

			print('training... step: {}/{} | loss: {:.4f} | acc: {:.2%}'.format(step+1, len(train_loader), loss, correct/len(labels)), end='\r')


		val_loss = 0
		val_correct = 0
		val_total = 0

		model.eval()
		feature_extractor.eval()
		with torch.no_grad():
			for step, batch in enumerate(val_loader):
				imgs, labels = batch
				imgs, labels = imgs.to(device), labels.to(device)

				features = feature_extractor(imgs)
				outputs = model(features)

				loss = criterion(outputs, labels)
				val_loss += loss.item()

				preds = torch.max(outputs, dim=1)[1]
				correct = (preds==labels).sum().item()
				val_correct += correct
				val_total += len(labels)

				print('validating... step: {}/{} | loss: {:.4f} | acc: {:.2%}'.format(step+1, len(val_loader), loss, correct/len(labels)), end='\r')

		print('epoch: {}/{} ---> train loss: {:.4f} | val loss: {:.4f} | train acc: {:.2%} | val acc: {:.2%}'.format(
			epoch+1, args.epochs, train_loss/len(train_loader), val_loss/len(val_loader), train_correct/train_total, val_correct/val_total))


		if val_correct/val_total > threshold:
			threshold = val_correct/val_total

			save_dir = os.path.dirname(args.model_path)
			if save_dir != '':
				os.makedirs(save_dir, exist_ok=True)

			torch.save(model.state_dict(), args.model_path)
			torch.save(feature_extractor.state_dict(), args.backbone_model_path)
			print('------- save model -------')










def main(args):
	fix_seed(args.seed)
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	feature_extractor = models.resnet50(pretrained=False)
	#feature_extractor.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
	feature_extractor = feature_extractor.to(device)

	transform = {
		'train': T.Compose([
			T.RandomResizedCrop(128, scale=(0.5, 1.0)),
			T.RandomHorizontalFlip(p=20),
			T.RandomGrayscale(p=0.1),
			T.ToTensor(),
			T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			]),
		'val': T.Compose([
			T.Resize((128, 128)),
			T.ToTensor(),
			T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			]),
	}


	train(args, feature_extractor, transform, device)






def argument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_path', default='../hw4_data/office/train', type=str)
	parser.add_argument('--val_path', default='../hw4_data/office/val', type=str)
	parser.add_argument('--pretrained', default='./ckpt/pretrained_resnet50.pth', type=str)
	parser.add_argument('--fix_backbone', action='store_true')
	parser.add_argument('--model_path', default='./ckpt/p2_model.pth', type=str)
	parser.add_argument('--backbone_model_path', default='./ckpt/p2_extractor.pth', type=str)
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--threshold', default=0.36, type=float)
	parser.add_argument('--device', default='cuda:0', type=str)
	parser.add_argument('--seed', default=123, type=int)


	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = argument()
	main(args)

