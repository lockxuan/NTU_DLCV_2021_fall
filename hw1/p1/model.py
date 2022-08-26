import torch
import torch.nn as nn
import torchvision.models






class p1_model(nn.Module):
	"""docstring for p1_model"""
	def __init__(self):
		super(p1_model, self).__init__()
		self.resnet = torchvision.models.resnet50(pretrained=True)
		self.out = nn.Sequential(
			nn.Linear(2048, 512),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512, 50)
			)
		self.resnet.fc = self.out


	def forward(self, x):
		x = self.resnet(x)

		return x


if __name__ == '__main__':
	print(p1_model())

		
		