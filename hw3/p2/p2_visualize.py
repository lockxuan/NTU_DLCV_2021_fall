import torch
import torch.nn as nn
import glob
import os
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

from transformers import BertTokenizer


import argparse





def LoadData(caption_path, imgs_path):
	pass


def create_caption_and_mask(start_token, max_length):
	caption_template = torch.zeros((1, max_length), dtype=torch.long)
	mask_template = torch.ones((1, max_length), dtype=torch.bool)

	caption_template[:, 0] = start_token
	mask_template[:, 0] = False

	return caption_template, mask_template


def interpolate_resize(heatmap, h, w):
	heatmap=heatmap.unsqueeze(0).unsqueeze(0)
	heatmap=torch.nn.functional.interpolate(heatmap, size=(h, w), mode='bicubic', align_corners=False)
	heatmap=heatmap.squeeze(0).squeeze(0)

	return heatmap

def main(args):
	invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
										std = [ 1/0.5, 1/0.5, 1/0.5 ]),
							T.Normalize(mean = [ -0.5, -0.5, -0.5 ],
										std = [ 1., 1., 1. ]),
						])
	Trans = T.Compose([
		T.ToTensor(),
		T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])

	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
	end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

	model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True).to(device)

	os.makedirs(args.output_path, exist_ok=True)

	imgs_path = glob.glob(os.path.join(args.val_path, '*.jpg'))

	for path in imgs_path:
		image = Image.open(path)
		image = Trans(image).unsqueeze(0).to(device)
		_, _, h, w = image.shape


		features = []
		def hook(module, input, output):
			features.append(output[1].clone().cpu().detach())

		handel = model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(hook)

		caption, cap_mask = create_caption_and_mask(start_token, 128)
		caption, cap_mask = caption.to(device), cap_mask.to(device)
		
		model.eval()
		for i in range(127):
			predictions = model(image, caption, cap_mask)
			predictions = predictions[:, i, :]
			predicted_id = torch.argmax(predictions, axis=-1)

			if predicted_id[0] == 102:
				caption[:, i+1] = predicted_id[0]
				break

			caption[:, i+1] = predicted_id[0]
			cap_mask[:, i+1] = False
		handel.remove()
		caption = caption.cpu()

		image = image.cpu()
		plt.figure(figsize=(8,8))
		plt.subplot(3, 5, 1)
		plt.title('[CLS]')
		plt.imshow(invTrans(image).squeeze(0).permute(1,2,0))
		plt.axis('off')
		for i in range(len(features)):
			r_h, r_w = math.ceil(h/16), math.ceil(w/16)
			heatmap = features[i][0][i].reshape((r_h, r_w))
			heatmap = interpolate_resize(heatmap, h, w)
			
			plt.subplot(3, 5, i+2)
			plt.title(tokenizer.decode(caption[0][i+1].tolist()))
			plt.imshow(invTrans(image).squeeze(0).permute(1,2,0))
			plt.imshow(heatmap, alpha=0.7, cmap='rainbow')
			plt.axis('off')
		plt.tight_layout()
		#plt.show()
		plt.savefig(os.path.join(args.output_path, path.split('/')[-1].replace('jpg', 'png')))
		
	





if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--val_path', type=str, default='../hw3_data/p2_data/images')
	parser.add_argument("--output_path", default='./')
	parser.add_argument('--seed', type=int, default=10)
	parser.add_argument('--device', type=str, default='cuda:0')

	args = parser.parse_args()

	main(args)




