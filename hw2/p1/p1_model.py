import torch
import torch.nn as nn
import torchvision






class DCGAN_G(nn.Module):
	"""docstring for DCGAN_G"""
	def __init__(self, G_in_size):
		super(DCGAN_G, self).__init__()

		# 1024 * 4 * 4
		self.conv1 = nn.Sequential(
			nn.ConvTranspose2d(G_in_size, 1024, 4, 1, 0, bias=False),
			nn.BatchNorm2d(1024),
			nn.ReLU()
			)

		# 512 * 8 * 8
		self.conv2 = nn.Sequential(
			nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU()
			)

		# 256 * 16 * 16
		self.conv3 = nn.Sequential(
			nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU()
			)

		# 128 * 32 * 32
		self.conv4 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU()
			)

		# 3 * 32 * 32
		self.conv5 = nn.Sequential(
			nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
			nn.Tanh()
			)



	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)

		return x



class DCGAN_D(nn.Module):
	"""docstring for DCGAN_D"""
	def __init__(self):
		super(DCGAN_D, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			)

		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 128, 4, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
			)

		self.conv3 = nn.Sequential(
			nn.Conv2d(128, 256, 4, 2, 1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),
			)

		self.conv4 = nn.Sequential(
			nn.Conv2d(256, 512, 4, 2, 1, bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True),
			)

		self.conv5 = nn.Sequential(
			nn.Conv2d(512, 1, 4, 1, 0, bias=False),
			nn.Sigmoid(),
			)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)

		return x.view(-1)





# Weights
def weights_init(m):
	classname = m.__class__.__name__

	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
	G = DCGAN_G(100)
	G.apply(weights_init)

	D = DCGAN_D()
	D.apply(weights_init)

	print(D)
	"""
	noise = torch.rand(8, 128, 1, 1)
	true_label = torch.ones(8, dtype=torch.float)
	fake_label = torch.zeros(8, dtype=torch.float)

	print(true_label)
	print(fake_label)
	out_G = G(noise)
	out_D = D(out_G)
	print(out_D)

	criterion = nn.BCELoss()
	print(criterion(out_D, fake_label))

	torchvision.utils.save_image(out_G.data, "test.png", normalize=True)
	"""


		
		