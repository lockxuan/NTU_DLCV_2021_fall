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

import os

from p2_model import ACGAN_G_2, ACGAN_D, weights_init
from p2_dataset import p2_dataloader, eval_dataset






def fix_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)


class Classifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 4 * 4, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1) # flatten all dimensions except batch
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


def inference(model_G, save_dir, device, num_photos=1000, hidden_dim=110):
	invTrans = T.Compose([ T.Resize(28),
							T.Normalize(mean = [ 0., 0., 0. ],
										std = [ 1/0.5, 1/0.5, 1/0.5 ]),
							T.Normalize(mean = [ -0.5, -0.5, -0.5 ],
										std = [ 1., 1., 1. ]),
						])

	fixed_noise = torch.randn(1000, hidden_dim, 1, 1)
	fake_noise = np.random.normal(0, 1, (1000, hidden_dim))
	fake_onehot = np.zeros((1000, 10))
	for i in range(10):
		fake_classes = np.ones(100, dtype=int)*i
		fake_onehot[np.arange(i*100, (i+1)*100), fake_classes] = 1
		fake_noise[np.arange(i*100, (i+1)*100), :10] = fake_onehot[np.arange(i*100, (i+1)*100)]

	fake_noise = (torch.from_numpy(fake_noise))
	fixed_noise.data.copy_(fake_noise.view(1000, hidden_dim, 1, 1))
	fixed_noise = fixed_noise.to(device)

	
	outputs = model_G(fixed_noise)
	outputs = invTrans(outputs)
	
	os.makedirs(save_dir, exist_ok=True)
	for i in range(10):
		for j, img in enumerate(outputs[i*100:(i+1)*100]):
			save_path = os.path.join(save_dir, str(i)+'_{0:03d}.png'.format(j+1))
			save_image(img, save_path)


def train(model_D, model_G, train_lodaer, criterion_tf, criterion_class, optimizer_D, optimizer_G, device, hidden_dim=110, save_dir='./example'):
	eval_transform = T.Compose([
		T.ToTensor(),
		T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5 )),
		])
	classifier = Classifier()
	state = torch.load("./p2/Classifier.pth", map_location = device)
	classifier.load_state_dict(state['state_dict'])
	classifier = classifier.to(device)

	fixed_noise = torch.randn(1000, hidden_dim, 1, 1)
	fake_noise = np.random.normal(0, 1, (1000, hidden_dim))
	fake_onehot = np.zeros((1000, 10))
	for i in range(10):
		fake_classes = np.ones(100, dtype=int)*i
		fake_onehot[np.arange(i*100, (i+1)*100), fake_classes] = 1
		fake_noise[np.arange(i*100, (i+1)*100), :10] = fake_onehot[np.arange(i*100, (i+1)*100)]

	fake_noise = (torch.from_numpy(fake_noise))
	fixed_noise.data.copy_(fake_noise.view(1000, hidden_dim, 1, 1))
	fixed_noise = fixed_noise.to(device)

	threshold = 0.7
	weight = 1

	for epoch in range(40):

		model_D.train()
		model_G.train()
		epoch_real_acc = 0
		epoch_fake_acc = 0
		total_real = 0
		total_fake = 0
		for step, batch in enumerate(train_lodaer):
			real_imgs, real_classes = batch
			real_imgs, real_classes = real_imgs.to(device), real_classes.to(device)

			real_label = torch.ones(real_imgs.size(0), dtype=torch.float).to(device)
			fake_label = torch.zeros(real_imgs.size(0), dtype=torch.float).to(device)

			### ******************** ###
			###	update Discriminator ###
			### ******************** ###
			model_D.zero_grad()

			out_rf, out_classes = model_D(real_imgs)

			preds = torch.max(out_classes, dim=1)[1]
			real_acc = (preds == real_classes).sum().item()
			epoch_real_acc += real_acc
			total_real += len(real_classes)
			
			loss_d_real = criterion_tf(out_rf, real_label) + weight*criterion_class(out_classes, real_classes)
			loss_d_real.backward()

			### make noise
			noise = torch.randn(real_imgs.size(0), hidden_dim, 1, 1)
			fake_noise = np.random.normal(0, 1, (real_imgs.size(0), hidden_dim))
			fake_classes = np.random.randint(0, 10, real_imgs.size(0))
			fake_onehot = np.zeros((real_imgs.size(0), 10))
			fake_onehot[np.arange(real_imgs.size(0)), fake_classes] = 1
			fake_noise[np.arange(real_imgs.size(0)), :10] = fake_onehot[np.arange(real_imgs.size(0))]
			fake_noise = (torch.from_numpy(fake_noise))
			noise.data.copy_(fake_noise.view(real_imgs.size(0), hidden_dim, 1, 1))
			noise = noise.to(device)

			fake_classes = torch.from_numpy(fake_classes).to(device)

			fake_imgs = model_G(noise)

			out_rf, out_classes = model_D(fake_imgs.detach())

			preds = torch.max(out_classes, dim=1)[1]
			fake_acc = (preds == fake_classes).sum().item()
			epoch_fake_acc += fake_acc
			total_fake += len(fake_classes)

			
			loss_d_fake = criterion_tf(out_rf, fake_label) + weight*criterion_class(out_classes, fake_classes)
			loss_d_fake.backward()

			loss_d = loss_d_real + loss_d_fake
			optimizer_D.step()


			### **************** ###
			###	update Generator ###
			### **************** ###
			model_G.zero_grad()
			#noise.data.copy_(torch.randn(real_imgs.size(0), hidden_dim, 1, 1))

			#fake_imgs = model_G(noise)
			out_rf, out_classes = model_D(fake_imgs)
			
			loss_g = criterion_tf(out_rf, real_label) + weight*criterion_class(out_classes, fake_classes)
			loss_g.backward()

			optimizer_G.step()

		
			print('epoch: {} | step: {}/{} | loss D: {:4f} | loss G: {:4f} | real acc: {:2%} | fake acc: {:2%}'.format(
				epoch+1, step+1, len(train_lodaer), loss_d, loss_g, real_acc/len(real_classes), fake_acc/len(fake_classes)), end='\r')
		print(' '*130, end='\r')

			
		model_G.eval()
		inference(model_G, 'p2_inference', device)

		test_dataset = eval_dataset('./p2_inference', eval_transform)
		eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

		total_correct = 0
		total_img = 0
		for step, batch in enumerate(eval_loader):
		    imgs, labels = batch
		    imgs, labels = imgs.to(device), labels.to(device)

		    outputs = classifier(imgs)

		    preds = torch.max(outputs, dim=1)[1]
		    correct = (preds == labels).sum()
		    total_correct += correct
		    total_img += len(labels)


		print('epoch: {} | real acc: {:2%} | fake acc: {:2%} | classifier acc: {:2%}'.format(
			epoch+1, epoch_real_acc/total_real, epoch_fake_acc/total_fake, total_correct/total_img))
		
		if total_correct/total_img > threshold:
			threshold = total_correct/total_img
			torch.save(model_D.state_dict(), './p2_net_D.pth')
			torch.save(model_G.state_dict(), './p2_net_G.pth')
		
	print('best acc: {}'.format(str(threshold)))









def main(args):
	fix_seed(seed)

	transform = T.Compose([
		#T.Resize(64),
		T.ToTensor(),
		T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5 )),
		])
	train_lodaer = p2_dataloader(args.img_folder, args.csv_path, train=True, batch_size=64, transform=transform)

	device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
	model_G = ACGAN_G_2(110).to(device)
	model_D = ACGAN_D().to(device)

	model_G.apply(weights_init)
	model_D.apply(weights_init)

	criterion_tf = nn.BCELoss().to(device)
	criterion_class = nn.NLLLoss().to(device)

	optimizer_G = optim.Adam(model_G.parameters(), lr=2e-4, betas=(0.5, 0.999))
	optimizer_D = optim.Adam(model_D.parameters(), lr=2e-4, betas=(0.5, 0.999))


	train(model_D, model_G, train_lodaer, criterion_tf, criterion_class, optimizer_D, optimizer_G, device)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--img_folder", default='./hw2_data/digits/mnistm/train')
	parser.add_argument("--csv_path", default='./hw2_data/digits/mnistm/train.csv')
	parser.add_argument("--seed", type=int, default=2022)

	args = parser.parse_args()
	main(args)



