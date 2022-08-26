import torch
import torch.nn as nn
import torchvision
import torchvision.models as models








class Extractor(nn.Module):
	"""docstring for Extractor"""
	def __init__(self):
		super(Extractor, self).__init__()

		self.resnet = nn.Sequential(
				*list(models.resnet18(pretrained=False).children())[:-1]
			)



	def forward(self, x):
		x = self.resnet(x).view(-1, 512)

		return x



class Classifier(nn.Module):
	"""docstring for Classifier"""
	def __init__(self):
		super(Classifier, self).__init__()
		
		self.fc = nn.Sequential(
			nn.Linear(512, 10),
			#nn.ReLU(),
			#nn.Dropout(0.5),
			#nn.Linear(256, 10)
			)

	def forward(self, x):
		x = self.fc(x)

		return x


class Discriminator(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self):
		super(Discriminator, self).__init__()
		
		self.fc = nn.Sequential(
			nn.Linear(512, 512),
			#nn.BatchNorm1d(512),
			nn.LeakyReLU(0.2),
			nn.Linear(512, 128),
			#nn.BatchNorm1d(128),
			nn.LeakyReLU(0.2),
			nn.Linear(128, 1),
			#nn.Sigmoid()
			)

	def forward(self, x):
		x = self.fc(x)

		return x
		




if __name__ == '__main__':
	inputs = torch.rand(16, 3, 28, 28)
	model = Extractor()
	model(inputs)





