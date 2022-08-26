import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np






class FCN32s(nn.Module):
	"""docstring for FCN32s"""
	def __init__(self):
		super(FCN32s, self).__init__()
		self.vgg = models.vgg16(pretrained=True).features

		self.conv6 = nn.Sequential(
			nn.Conv2d(512, 4096, 1),
			nn.ReLU(inplace=True),
			nn.Dropout()
			)
		
		self.conv7 = nn.Sequential(
			nn.Conv2d(4096, 4096, 1),
			nn.ReLU(inplace=True),
			nn.Dropout()
			)
		self.score = nn.Conv2d(4096, 7, 1)
		self.upsample =  nn.ConvTranspose2d(7, 7, 64, stride=32, bias=False)

	def forward(self, x):
		out = self.vgg(x)
		out = self.conv6(out)
		out = self.conv7(out)
		out = self.score(out)
		out = self.upsample(out)
		out = out[:, :, 16:16+x.shape[2], 16:16+x.shape[3]]

		return out



def bilinear_kernel(in_channels, out_channels, kernel_size):
	factor = (kernel_size + 1) // 2
	if kernel_size % 2 == 1:
		center = factor - 1
	else:
		center = factor - 0.5
	og = np.ogrid[:kernel_size, :kernel_size]
	filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
	weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
	weight[range(in_channels), range(out_channels), :, :] = filt
	return torch.from_numpy(weight)

class FCN8sVGG(nn.Module):
	"""docstring for FCN8sVGG"""
	def __init__(self):
		super(FCN8sVGG, self).__init__()
		vgg = models.vgg16(pretrained=True).features
		self.vgg1 = nn.Sequential(*list(vgg.children())[:17])
		self.vgg2 = nn.Sequential(*list(vgg.children())[17:24])
		self.vgg3 = nn.Sequential(*list(vgg.children())[24:])

		self.fc = nn.Sequential(
			nn.Conv2d(512, 4096, 1),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Conv2d(4096, 4096, 1),
			nn.ReLU(inplace=True),
			nn.Dropout()			
			)
		
		self.score1 = nn.Conv2d(256, 7, 1)
		self.score2 = nn.Conv2d(512, 7, 1)
		self.score3 = nn.Conv2d(4096, 7, 1)

		self.upsample8 = nn.ConvTranspose2d(7, 7, 16, 8, 4, bias=False)
		self.upsample8.weight.data = bilinear_kernel(7, 7, 16)
		self.upsample4 = nn.ConvTranspose2d(7, 7, 4, 2, 1, bias=False)
		self.upsample4.weight.data = bilinear_kernel(7, 7, 4)
		self.upsample2 = nn.ConvTranspose2d(7, 7, 4, 2, 1, bias=False)
		self.upsample2.weight.data = bilinear_kernel(7, 7, 4)


	def forward(self, x):
		x1 = self.vgg1(x)
		x2 = self.vgg2(x1)
		x3 = self.vgg3(x2)
		x3 = self.fc(x3)

		x3 = self.score3(x3)
		x3 = self.upsample2(x3)

		x2 = self.score2(x2)
		x2 = x2 + x3
		x2 = self.upsample4(x2)

		x1 = self.score1(x1)
		out = x1 + x2

		out = self.upsample8(out)

		return out




if __name__ == '__main__':
	model = FCN8sVGG()
	print(model)
	print(sum(p.numel() for p in model.parameters()))
		






