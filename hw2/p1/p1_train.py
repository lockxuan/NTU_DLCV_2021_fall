import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import save_image
import torchvision
import argparse
import random

import os

from p1_model import DCGAN_G, DCGAN_D, weights_init
from p1_dataset import p1_dataloader

from pytorch_fid import fid_score






def fix_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)


def train(model_D, model_G, train_lodaer, criterion, optimizer_D, optimizer_G, device, hidden_dim=100, save_dir='./example'):
	invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
										std = [ 1/0.5, 1/0.5, 1/0.5 ]),
							T.Normalize(mean = [ -0.5, -0.5, -0.5 ],
										std = [ 1., 1., 1. ]),
						])
	fixed_noise = torch.randn(1000, hidden_dim, 1, 1).to(device)
	threshold = 40

	for epoch in range(110):
		model_D.train()
		model_G.train()
		for step, batch in enumerate(train_lodaer):
			real_imgs = batch.to(device)
			true_label = torch.ones(real_imgs.size(0), dtype=torch.float).to(device)
			fake_label = torch.zeros(real_imgs.size(0), dtype=torch.float).to(device)

			### ******************** ###
			###	update Discriminator ###
			### ******************** ###
			model_D.zero_grad()

			out = model_D(real_imgs)
			
			loss_d_real = criterion(out, true_label)
			loss_d_real.backward()

			noise = torch.randn(real_imgs.size(0), hidden_dim, 1, 1).to(device)
			fake_imgs = model_G(noise)
			out = model_D(fake_imgs.detach())
			
			loss_d_fake = criterion(out, fake_label)
			loss_d_fake.backward()

			loss_d = loss_d_real + loss_d_fake
			optimizer_D.step()


			### **************** ###
			###	update Generator ###
			### **************** ###
			model_G.zero_grad()
			noise.data.copy_(torch.randn(real_imgs.size(0), hidden_dim, 1, 1))

			fake_imgs = model_G(noise)
			out = model_D(fake_imgs)
			
			loss_g = criterion(out, true_label)
			loss_g.backward()

			optimizer_G.step()

		
			print('epoch: {} | step: {}/{} | loss D: {:4f} | loss G: {:4f}'.format(
				epoch+1, step+1, len(train_lodaer), loss_d, loss_g), end='\r')

			
		model_G.eval()
		outputs = model_G(fixed_noise)

		save_path = os.path.join('example_img')
		os.makedirs(save_path, exist_ok=True)
		torchvision.utils.save_image(outputs.data[:32], "%s/%s.png" % (save_path, str(epoch)), normalize=True)


		os.makedirs('inference', exist_ok=True)
		for i, img in enumerate(outputs):
			save_path = os.path.join('inference', '{0:04d}.png'.format(i))
			save_image(invTrans(img), save_path)

		fid = fid_score.calculate_fid_given_paths(['hw2_data/face/test','inference'], 64, device, 2048)
		print('epoch: {} | fid: {:4f}'.format(epoch+1, fid))

		if fid<threshold:
			threshold = fid
			torch.save(model_D.state_dict(), './net_D.pth')
			torch.save(model_G.state_dict(), './net_G.pth')
			print('----- save model at fid:', fid)

	print('best fid: {}'.format(str(threshold)))









def main(args):
	fix_seed(999)

	transform = T.Compose([
		#T.Resize(),
		T.ToTensor(),
		T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5 )),
		])
	train_lodaer = p1_dataloader(args.train_path, batch_size=128, transform=transform)

	device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
	model_G = DCGAN_G(100).to(device)
	model_D = DCGAN_D().to(device)

	model_G.apply(weights_init)
	model_D.apply(weights_init)

	criterion = nn.BCELoss().to(device)

	optimizer_G = optim.Adam(model_G.parameters(), lr=2e-4, betas=(0.5, 0.999))
	optimizer_D = optim.Adam(model_D.parameters(), lr=2e-4, betas=(0.5, 0.999))


	train(model_D, model_G, train_lodaer, criterion, optimizer_D, optimizer_G, device)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_path", default='./hw2_data/face/train')
	args = parser.parse_args()
	main(args)



