import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision
import argparse
import random
import numpy as np

import os

from p3_model import Extractor, Classifier
from p3_dataset import p3_dataloader, p3_dataset






def fix_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)



def train(extractor, classifier, train_loader, eval_loader, test_loader, criterion, optimizer, device, model_name):
	threshold = 1

	for epoch in range(20):

		extractor.train()
		classifier.train()

		train_loss = 0
		eval_loss = 0

		train_correct = 0
		eval_correct = 0

		train_len = 0
		eval_len = 0

		for step, batch in enumerate(train_loader):
			imgs, labels = batch
			imgs, labels = imgs.to(device), labels.to(device)

			optimizer.zero_grad()

			features = extractor(imgs)
			outputs = classifier(features)

			loss = criterion(outputs, labels)
			loss.backward()
			train_loss += loss.item()

			preds = torch.max(outputs, dim=1)[1]
			correct = (preds==labels).sum()
			train_correct += correct
			train_len += len(labels)

			optimizer.step()

			print('training... step: {}/{} | loss: {:.4f} | acc: {:.2%}'.format(step+1, len(train_loader), loss, correct/len(labels)), end='\r')

		extractor.eval()
		classifier.eval()
		with torch.no_grad():
			for step, batch in enumerate(eval_loader):
				imgs, labels = batch
				imgs, labels = imgs.to(device), labels.to(device)

				features = extractor(imgs)
				outputs = classifier(features)

				loss = criterion(outputs, labels)
				eval_loss += loss.item()

				preds = torch.max(outputs, dim=1)[1]
				correct = (preds==labels).sum()
				eval_correct += correct
				eval_len += len(labels)

				print('evaluate... step: {}/{} | loss: {:.4f} | acc: {:.2%}'.format(step+1, len(eval_loader), loss, correct/len(labels)), end='\r')
		
		print('epoch: {} | train loss: {:.4f} | eval loss: {:.4f} | train acc: {:.2%} | eval acc: {:.2%}'.format(
			epoch+1, train_loss/len(train_loader), eval_loss/len(eval_loader), train_correct/train_len, eval_correct/eval_len))

		
		#if eval_loss/len(eval_loader) + 2*(1 - eval_correct/eval_len) < threshold: # loss + 2*(1-acc)
		#	threshold = eval_loss/len(eval_loader) * (1 - eval_correct/eval_len)

	save_path = './ckpt'
	os.makedirs(save_path, exist_ok=True)

	torch.save(extractor.state_dict(), os.path.join(save_path, model_name+'_e.pth'))
	torch.save(classifier.state_dict(), os.path.join(save_path, model_name+'_c.pth'))
		
	#print('best threshold: {} \n'.format(str(threshold)))

	extractor.eval()
	classifier.eval()
	test_loss = 0
	test_correct = 0
	test_len = 0
	with torch.no_grad():
		for step, batch in enumerate(test_loader):
			imgs, labels = batch
			imgs, labels = imgs.to(device), labels.to(device)

			features = extractor(imgs)
			outputs = classifier(features)

			loss = criterion(outputs, labels)
			test_loss += loss.item()

			preds = torch.max(outputs, dim=1)[1]
			correct = (preds==labels).sum()
			test_correct += correct
			test_len += len(labels)
		print('testing result,  loss: {:.4f} | acc: {:.2%}'.format(test_loss/len(test_loader), test_correct/test_len))






def main(args):
	#fix_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	transform = T.Compose([
		T.Resize(28),
		T.Grayscale(3),
		T.ToTensor(),
		T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
		])
	train_dataset = p3_dataset(args.train_path, transform=transform)
	train_len = len(train_dataset)
	train_dataset, eval_dataset = random_split(train_dataset, [train_len-train_len//9, train_len//9])

	train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
	eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
	test_loader = p3_dataloader(args.test_path, batch_size=64, transform=transform)

	device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

	extractor = Extractor().to(device)
	classifier = Classifier().to(device)

	criterion = nn.CrossEntropyLoss().to(device)

	optimizer = optim.Adam(list(extractor.parameters()) + list(classifier.parameters()), lr=2e-4)


	train(extractor, classifier, train_loader, eval_loader, test_loader, criterion, optimizer, device, args.model_name)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_path", default='./hw2_data/digits/usps/train')
	parser.add_argument("--test_path", default='./hw2_data/digits/svhn/test')
	parser.add_argument("--seed", type=int, default=2022)
	parser.add_argument("--model_name", type=str, default='p3_usps_pre')

	args = parser.parse_args()
	main(args)



