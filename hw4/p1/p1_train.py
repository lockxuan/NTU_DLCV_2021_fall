import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import torchvision.transforms as T
import random
import numpy as np
import argparse

from p1_dataset import p1_dataset, p1_sampler
from p1_model import Convnet
from p1_utils import worker_init_fn, cal_distance, cal_accuracy


def init_seed(seed):
	torch.manual_seed(seed)
	#torch.backends.cudnn.deterministic = True
	#torch.backends.cudnn.benchmark = False
	random.seed(seed)
	np.random.seed(seed)



def train(args, transform, device):
	train_dataset = p1_dataset(args.train_path, transform['train'])
	val_dataset = p1_dataset(args.val_path, transform['val'])

	train_sampler = p1_sampler(train_dataset.labels,
					n_way=args.train_n_way, k_shot=args.k_shot, k_query=args.k_query)
	val_sampler = p1_sampler(val_dataset.labels,
					n_way=args.val_n_way, k_shot=args.k_shot, k_query=args.k_query)

	train_loader = DataLoader(train_dataset, num_workers=5, worker_init_fn=worker_init_fn,
		batch_sampler=train_sampler)
	val_loader = DataLoader(val_dataset, num_workers=5, worker_init_fn=worker_init_fn,
		batch_sampler=val_sampler)

	model = Convnet().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	criterion = nn.CrossEntropyLoss()

	threshold = args.threshold
	distance = cal_distance(method=args.distance, model=model)

	for epoch in range(args.epochs):
		train_loss = 0
		train_correct = 0
		train_total = 0

		model.train()
		for step, batch in enumerate(train_loader):
			imgs, raw_labels = batch
			imgs = imgs.to(device)

			suppout_num = args.train_n_way*args.k_shot
			support_set, query_set = imgs[:suppout_num], imgs[suppout_num:]

			prototype = model(support_set) # (n_way*k_shot, out_dim=64)
			prototype = prototype.reshape(args.train_n_way, args.k_shot, -1) # (n_way, k_shot, out_dim)
			prototype = prototype.mean(dim=1) # (n_way, out_dim)

			query = model(query_set) # (n_way*q_query, out_dim=64)
			outputs = distance(query, prototype)
			labels = torch.arange(args.train_n_way).repeat_interleave(args.k_query).long().to(device)

			loss = criterion(outputs, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			accuracy, correct, num_labels = cal_accuracy(outputs, labels)
			train_correct += correct
			train_total += num_labels
			train_loss += loss.item()

			print('training...  step: {}/{} | loss: {:.4f} | accuracy: {:.2%}'.format(
				step+1, train_sampler.total_batches, loss, accuracy), end='\r')
			
		
		val_loss = 0
		val_correct = 0
		val_total = 0

		model.eval()
		with torch.no_grad():
			for step, batch in enumerate(val_loader):
				imgs, _ = batch
				imgs = imgs.to(device)

				suppout_num = args.val_n_way*args.k_shot
				support_set, query_set = imgs[:suppout_num], imgs[suppout_num:]

				prototype = model(support_set) # (n_way*k_shot, out_dim=64)
				prototype = prototype.reshape(args.val_n_way, args.k_shot, -1) # (k_shot, n_way, out_dim)
				prototype = prototype.mean(dim=1) # (n_way, out_dim)

				query = model(query_set) # (n_way*q_query, out_dim=64)
				outputs = distance(query, prototype)
				labels = torch.arange(args.val_n_way).repeat_interleave(args.k_query).long().to(device)

				loss = criterion(outputs, labels)

				accuracy, correct, num_labels = cal_accuracy(outputs, labels)
				val_correct += correct
				val_total += num_labels
				val_loss += loss.item()

				print('validating...  step: {}/{} | loss: {:.4f} | accuracy: {:.2%}'.format(
					step+1, val_sampler.total_batches, loss, accuracy), end='\r')
				

		print('epoch: {}/{} | train loss: {:.4f} | val loss: {:.4f} | train acc: {:.2%} | val acc: {:.2%}'.format(
			epoch+1, args.epochs, train_loss/train_sampler.total_batches, val_loss/val_sampler.total_batches,
			train_correct/train_total, val_correct/val_total))

		if val_correct/val_total > threshold:
			threshold = val_correct/val_total

			save_dir = os.path.dirname(args.save_model_path)
			if save_dir != '':
				os.makedirs(save_dir, exist_ok=True)

			torch.save(model.state_dict(), args.save_model_path)

	print('final save val accuracy:', threshold)










def main(args):
	init_seed(args.seed)
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	transform = {
		'train': T.Compose([
				#T.Resize((84, 84)),
				T.RandomResizedCrop(84,scale=(0.5, 1.0)),
				T.RandomHorizontalFlip(p=20),
				T.RandomGrayscale(p=0.1),
				T.ToTensor(),
				T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			]),
		'val': T.Compose([
				T.Resize((84, 84)),
				T.ToTensor(),
				T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
	}


	train(args, transform, device)







if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_path', type=str, default='../hw4_data/mini/train')
	parser.add_argument('--val_path', type=str, default='../hw4_data/mini/val')
	parser.add_argument('--save_model_path', type=str, default='./ckpt/p1_model.pth')
	parser.add_argument('--train_n_way', type=int, default=15)
	parser.add_argument('--val_n_way', type=int, default=5)
	parser.add_argument('--k_shot', type=int, default=1)
	parser.add_argument('--k_query', type=int, default=15)
	parser.add_argument('--epochs', type=int, default=80)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--threshold', type=float, default=0.42)
	parser.add_argument('--distance', type=str, default='euclidean')
	parser.add_argument('--seed', type=int, default=2021)
	parser.add_argument('--device', type=str, default='cuda:0')
	args = parser.parse_args()
	main(args)






