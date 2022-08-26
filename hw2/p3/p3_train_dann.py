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

from p3_model import Extractor, Classifier, Discriminator
from p3_dataset import p3_dataloader, p3_dataset






def fix_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True



def train(extractor, classifier, discriminator, dataloaders, criterion_bce, criterion_ce, optimizer_e, optimizer_c, optimizer_d, device):
	best_acc = 0.72

	BATCHS = 25
	target_set = iter(dataloaders['target_train'])
	for epoch in range(BATCHS):

		extractor.train()
		classifier.train()
		discriminator.train()

		train_loss_d = 0
		train_loss_c = 0

		train_correct = 0

		train_len = 0

		tgt_step = 0
		

		p = (epoch+1) / BATCHS
		lamda = 2./(1+np.exp(-10.*p)) -1

		for step, batch in enumerate(dataloaders['source_train']):
			if tgt_step % len(dataloaders['target_train']) == 0:
				target_set = iter(dataloaders['target_train'])
			tgt_set = next(target_set)

			tgt_imgs, _ = tgt_set
			src_imgs, labels = batch

			src_imgs, labels, tgt_imgs = src_imgs.to(device), labels.to(device), tgt_imgs.to(device)

			d_src = torch.ones(src_imgs.size(0), 1).to(device)
			d_tgt = torch.zeros(tgt_imgs.size(0), 1).to(device)
			d_labels = torch.cat([d_src, d_tgt], dim=0)

			x_d = torch.cat((src_imgs, tgt_imgs), dim=0)
			features = extractor(x_d)
			y_d = discriminator(features.detach())

			loss_d = criterion_bce(y_d, d_labels)
			train_loss_d += loss_d.item()

			discriminator.zero_grad()
			loss_d.backward()
			optimizer_d.step()

			#########
			c_out = classifier(features[:src_imgs.size(0)])
			y_d = discriminator(features)

			loss_c = criterion_ce(c_out, labels)
			loss_d_ = criterion_bce(y_d, d_labels)
			loss_c = loss_c - 0.02*lamda*loss_d_
			train_loss_c += loss_c.item()

			extractor.zero_grad()
			classifier.zero_grad()
			discriminator.zero_grad()
			loss_c.backward()

			torch.nn.utils.clip_grad_norm_(extractor.parameters(), 0.001)

			optimizer_e.step()
			optimizer_c.step()


			preds = torch.max(c_out, dim=1)[1]
			correct = (preds==labels).sum()
			train_correct += correct
			train_len += len(labels)

			print('training... step: {}/{} | loss c: {:.4f} | loss d: {:.4f} | acc: {:.2%}'.format(
				step+1, len(dataloaders['source_train']), loss_c, loss_d, correct/len(labels)), end='\r')

			tgt_step += 1

		extractor.eval()
		classifier.eval()
		eval_loss_s = 0
		eval_correct_s = 0
		eval_len_s = 0
		eval_loss_t = 0
		eval_correct_t = 0
		eval_len_t = 0
		with torch.no_grad():
			for step, batch in enumerate(dataloaders['source_test']):
				imgs, labels = batch
				imgs, labels = imgs.to(device), labels.to(device)

				features = extractor(imgs)
				outputs = classifier(features)

				loss = criterion_ce(outputs, labels)
				eval_loss_s += loss.item()

				preds = torch.max(outputs, dim=1)[1]
				correct = (preds==labels).sum()
				eval_correct_s += correct
				eval_len_s += len(labels)

				print('evaluate... step: {}/{} | loss: {:.4f} | acc: {:.2%}'.format(
					step+1, len(dataloaders['source_test']), loss, correct/len(labels)), end='\r')

			for step, batch in enumerate(dataloaders['target_test']):
				imgs, labels = batch
				imgs, labels = imgs.to(device), labels.to(device)

				features = extractor(imgs)
				outputs = classifier(features)

				loss = criterion_ce(outputs, labels)
				eval_loss_t += loss.item()

				preds = torch.max(outputs, dim=1)[1]
				correct = (preds==labels).sum()
				eval_correct_t += correct
				eval_len_t += len(labels)

				print('evaluate... step: {}/{} | loss: {:.4f} | acc: {:.2%}'.format(
					step+1, len(dataloaders['target_test']), loss, correct/len(labels)), end='\r')


		if eval_correct_t/eval_len_t >= best_acc:
			best_acc = eval_correct_t/eval_len_t
			save_dir = './ckpt'
			os.makedirs(save_dir, exist_ok=True)
			torch.save(extractor.state_dict(), os.path.join(save_dir, './p3_m2u_e.pth'))
			torch.save(classifier.state_dict(),  os.path.join(save_dir, './p3_m2u_c.pth'))
			#torch.save(discriminator.state_dict(),  os.path.join(save_dir, './p3_u2s_d.pth'))

			print('save at acc:', best_acc.item())

		
		print('epoch: {} | train loss class/domain: {:.4f} / {:.4f} | eval loss class: {:.4f} | train acc: {:.2%} | eval acc: {:.2%}'.format(
			epoch+1, train_loss_c/len(dataloaders['source_train']), train_loss_d/len(dataloaders['source_train']),
			eval_loss_s/len(dataloaders['source_test']), train_correct/train_len, eval_correct_s/eval_len_s))

		print('epoch: {} | test loss: {:.4f} | test acc: {} \n'.format(
			epoch+1, eval_loss_t/len(dataloaders['target_test']), eval_correct_t/eval_len_t))







def main(args):
	fix_seed(args.seed)

	transform = T.Compose([
		T.Resize(28),
		T.ColorJitter(),
		T.RandomRotation(15),
		T.Grayscale(3),
		T.ToTensor(),
		T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
		])
	eval_transform = T.Compose([
		T.Resize(28),
		#T.RandomRotation(15),
		T.Grayscale(3),
		T.ToTensor(),
		T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
		])

	source_dataset = p3_dataset(args.source_path, transform=transform)
	len_source = len(source_dataset)
	source_dataset_t, source_dataset_e = random_split(source_dataset, [len_source-len_source//9, len_source//9])

	target_dataset = p3_dataset(args.target_path, transform=eval_transform)
	len_target = len(target_dataset)
	target_dataset_t, target_dataset_e = random_split(target_dataset, [len_target-len_target//9, len_target//9])

	source_loader = DataLoader(source_dataset_t, batch_size=64, shuffle=True)
	source_loader_test = DataLoader(source_dataset_e, batch_size=64, shuffle=False)
	target_loader = DataLoader(target_dataset_t, batch_size=64, shuffle=True)
	target_loader_test = DataLoader(target_dataset_e, batch_size=64, shuffle=False)

	dataloaders = {
			'source_train': source_loader,
			'source_test': source_loader_test,
			'target_train': target_loader,
			'target_test': target_loader_test
	}

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	extractor = Extractor().to(device)
	classifier = Classifier().to(device)
	discriminator = Discriminator().to(device)


	criterion_bce = nn.BCEWithLogitsLoss().to(device)
	criterion_ce = nn.CrossEntropyLoss().to(device)

	optimizer_e = optim.Adam(extractor.parameters(), lr=1e-3)
	optimizer_c = optim.Adam(classifier.parameters(), lr=1e-3)
	optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-3)


	train(extractor, classifier, discriminator, dataloaders, criterion_bce, criterion_ce, optimizer_e, optimizer_c, optimizer_d, device)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path", default='./hw2_data/digits/mnistm/train')
	parser.add_argument("--target_path", default='./hw2_data/digits/usps/train')
	parser.add_argument("--seed", type=int, default=2022)

	args = parser.parse_args()
	main(args)



