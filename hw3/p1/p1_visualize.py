import matplotlib.pyplot as plt
from pytorch_pretrained_vit import ViT
import torch
import torch.nn as nn
from torch.nn import functional as F

from p1_model import p1_model

from PIL import Image
import torchvision.transforms as T





def vis_positional_embedding(model):

	pos_embedding = model.vit.positional_embedding.pos_embedding

	fig = plt.figure(figsize=(8, 8))

	for i in range(1, pos_embedding.shape[1]):
		sim = F.cosine_similarity(pos_embedding[0, i:i+1], pos_embedding[0, 1:], dim=1)
		sim = sim.reshape((24, 24)).detach().cpu().numpy()
		ax = fig.add_subplot(24, 24, i)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		ax.imshow(sim)
	plt.show()


def vis_attention(model):
	invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
										std = [ 1/0.5, 1/0.5, 1/0.5 ]),
							T.Normalize(mean = [ -0.5, -0.5, -0.5 ],
										std = [ 1., 1., 1. ]),
						])
	destination = ['../hw3_data/p1_data/val/26_5064.jpg', '../hw3_data/p1_data/val/29_4718.jpg', '../hw3_data/p1_data/val/31_4838.jpg']

	img = Image.open(destination[2]).convert('RGB')
	img = T.Resize((384, 384))(img)
	img = T.ToTensor()(img)
	img = T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])(img)
	img = img.unsqueeze(0)
	model(img)
	#print(model.vit.transformer.blocks[11].attn.scores.shape) #b, 12, 577, 577
	attention_matrix = model.vit.transformer.blocks[11].attn.scores
	attention_matrix = torch.mean(attention_matrix, dim=1) #b, 577, 577

	heatmap = attention_matrix[0][0][1:].reshape((24, 24)).detach()#.cpu().numpy()
	heatmap = heatmap.unsqueeze(0).unsqueeze(0)
	heatmap=torch.nn.functional.interpolate(heatmap, size=(384, 384), mode='bicubic', align_corners=False)
	heatmap = heatmap.squeeze(0).squeeze(0)

	plt.figure(8)
	plt.subplot(221)
	plt.imshow(invTrans(img).squeeze(0).permute(1,2,0))
	plt.subplot(222)
	plt.imshow(invTrans(img).squeeze(0).permute(1,2,0))
	plt.imshow(heatmap, alpha=0.6, cmap='rainbow')
	plt.show()






def main():
	model = p1_model()
	model.load_state_dict(torch.load('ckpt/fine_adam_94.pth', map_location='cpu'))

	vis_positional_embedding(model)
	#vis_attention(model)

if __name__ == '__main__':
	main()
