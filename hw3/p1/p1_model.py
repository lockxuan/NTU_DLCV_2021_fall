from pytorch_pretrained_vit import ViT
import torch
import torch.nn as nn
from torch.nn import functional as F






class p1_model(nn.Module):
	"""docstring for p1_model"""
	def __init__(self):
		super(p1_model, self).__init__()
		
		model_name = 'B_16_imagenet1k'
		self.vit = ViT(model_name, pretrained=True)

		self.fc = nn.Sequential(
			nn.Linear(1000, 37),
			)


	def forward(self, x):
		x = self.vit(x)
		x = self.fc(x)

		return x

		






if __name__ == '__main__':
	img = torch.randn(2,3,224,224)
	model = p1_model()
	#model.load_state_dict(torch.load('ckpt/adam_16.pth', map_location='cpu'))

	#print(model.vit.positional_embedding.pos_embedding)
	#print(model.vit.image_size)
	print(model(img).shape)
		


