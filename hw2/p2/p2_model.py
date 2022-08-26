import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F







class ACGAN_D(nn.Module):
	"""docstring for ACGAN_D"""
	def __init__(self):
		super(ACGAN_D, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 28, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.5)
			)

		self.conv2 = nn.Sequential(
			nn.Conv2d(28, 56, 4, 2, 1, bias=False),
			nn.BatchNorm2d(56),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.5)
			)

		self.conv3 = nn.Sequential(
			nn.Conv2d(56, 112, 3, 2, 1, bias=False),
			nn.BatchNorm2d(112),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.5)
			)


		self.conv4 = nn.Sequential(
			nn.Conv2d(112, 224, 4, 1, 0, bias=False),
			)


		self.fc_real_fake = nn.Linear(224, 1)
		self.fc_class = nn.Linear(224, 10)

		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x=x.view(-1, 224)
		real_fake = self.sigmoid(self.fc_real_fake(x)).squeeze(1)
		classes = self.softmax(self.fc_class(x))

		return real_fake, classes


class ACGAN_G(nn.Module):
	"""docstring for ACGAN_G"""
	def __init__(self, G_in_size):
		super(ACGAN_G, self).__init__()

		self.conv1 = nn.Sequential(
			nn.ConvTranspose2d(G_in_size, 28*4, 4, 1, 0, bias=False),
			nn.BatchNorm2d(28*4),
			nn.ReLU()
			)

		self.conv2 = nn.Sequential(
			nn.ConvTranspose2d(28*4, 28*2, 3, 2, 1, bias=False),
			nn.BatchNorm2d(28*2),
			nn.ReLU()
			)

		self.conv3 = nn.Sequential(
			nn.ConvTranspose2d(28*2, 28, 4, 2, 1, bias=False),
			nn.BatchNorm2d(28),
			nn.ReLU()
			)

		self.conv4 = nn.Sequential(
			nn.ConvTranspose2d(28, 3, 4, 2, 1, bias=False),
			nn.Tanh()
			)



	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)

		return x



def weights_init(m):
	classname = m.__class__.__name__

	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
	print(ACGAN_G(110))


