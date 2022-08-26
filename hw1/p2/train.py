from dataset import p2_dataloader
from model import FCN32s, FCN8sVGG

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import argparse

import numpy as np
import math





def mean_iou_score(pred, labels):
	'''
	Compute mean IoU score over 6 classes
	'''
	mean_iou = 0
	for i in range(6):
		tp_fp = np.sum(pred == i)
		tp_fn = np.sum(labels == i)
		tp = np.sum((pred == i) * (labels == i))
		iou = tp / (tp_fp + tp_fn - tp)
		#if not math.isnan(iou):
		mean_iou += iou / 6
		#print('class #%d : %1.5f'%(i, iou))

	#print('\nmean_iou: %f\n' % mean_iou)

	return mean_iou


def save_model(model, save):
	torch.save(model.state_dict(), save)


def train(EPOCHS, model, train_loader, test_loader, criterion, optimizer, device, save=None):
	#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

	global_threshlod = 0.25
	best_train_iou = 0
	best_train_loss = 0
	best_test_iou = 0
	best_test_loss = 0
	best_epoch = 0


	for epoch in range(EPOCHS):
		model.train()

		train_loss = 0
		train_iou = 0
		preds = []
		masks = []
		for step, batch in enumerate(train_loader):
			sat, mask = batch
			sat, mask = sat.to(device), mask.to(device)
			optimizer.zero_grad()

			output = model(sat)

			loss = criterion(output, mask)
			train_loss += loss.item()
			loss.backward()

			optimizer.step()

			pred = torch.max(output, dim=1)[1]
			#mean_iou = mean_iou_score(pred, mask)
			#train_iou += mean_iou

			for p, m in zip(pred.detach().cpu().numpy(), mask.detach().cpu().numpy()):
				preds.append(p)
				masks.append(m)

			#print('training progress: {}/{} | iou: {}'.format(step+1, len(train_loader), mean_iou), end='\r')
			print('training progress: {}/{}'.format(step+1, len(train_loader)), end='\r')
		print('', end='\r')

		train_loss /= len(train_loader)
		#train_iou /= (step+1)
		train_iou = mean_iou_score(np.array(preds), np.array(masks))
		#train_iou = mean_iou_score(preds, masks)

		test_loss, test_iou = evaluate(model, test_loader, criterion, device)

		#scheduler.step(test_loss)
		scheduler.step()

		print('epoch: {}/{} | train loss: {:3f} | train iou: {:4f} | val loss: {:3f} | val iou: {:4f}'.format(
			epoch+1, EPOCHS, train_loss, train_iou, test_loss, test_iou))

		if (1-test_iou)*test_loss<global_threshlod and save:
			global_threshlod = (1-test_iou)*test_loss
			best_train_iou = train_iou
			best_train_loss = train_loss
			best_test_iou = test_iou
			best_test_loss = test_loss
			best_epoch = epoch+1
			save_model(model, save)
			#print('save at epoch: {}'.format(epoch+1))

		if (epoch+1)==3:
			save_model(model, 'p2_model_3.pth')
		elif (epoch+1)==10:
			save_model(model, 'p2_model_10.pth')

	#save_model(model, 'p2_last_model_res.pth')


	print('\nbest epoch: {} | train loss: {:3f} | train iou: {:4f} | val loss: {:3f} | val iou: {:4f}'.format(
			best_epoch, best_train_loss, best_train_iou, best_test_loss, best_test_iou))


def evaluate(model, test_loader, criterion, device):
	model.eval()

	test_loss = 0
	test_iou = 0
	mean_iou = 0

	preds = []
	masks = []

	with torch.no_grad():
		for step, batch in enumerate(test_loader):
			sat, mask = batch
			sat, mask = sat.to(device), mask.to(device)

			output = model(sat)

			loss = criterion(output, mask)
			test_loss += loss.item()

			pred = torch.max(output, dim=1)[1]
			#mean_iou = mean_iou_score(pred, mask)
			#test_iou += mean_iou
			for p, m in zip(pred.detach().cpu().numpy(), mask.detach().cpu().numpy()):
				preds.append(p)
				masks.append(m)

			#print('testing progress: {}/{} | iou: {}'.format(step+1, len(test_loader), mean_iou), end='\r')
			print('testing progress: {}/{}'.format(step+1, len(test_loader)), end='\r')
		print('', end='\r')
	test_loss /= len(test_loader)
	#test_iou /= (step+1)
	test_iou = mean_iou_score(np.array(preds), np.array(masks))


	return test_loss, test_iou







def main(args):
	transform = {'train': T.Compose([
			T.ToTensor(),
			T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
		]),
		'val': T.Compose([
			T.ToTensor(),
			T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
		])
	}


	lr = 0.001
	EPOCHS = 30
	device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

	model = FCN8sVGG().to(device)
	train_loader = p2_dataloader(args.train_path, batch_size=8, train=True, shuffle=True, transform=transform['train'])
	test_loader = p2_dataloader(args.test_path, batch_size=8, train=True, shuffle=False, transform=transform['val'])

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)

	cpt_path = 'p2_model.pth'
	train(EPOCHS, model, train_loader, test_loader, criterion, optimizer, device, cpt_path)







if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_path", default='./hw1_data/p2_data/train')
	parser.add_argument("--test_path", default='./hw1_data/p2_data/validation')
	args = parser.parse_args()
	main(args)



