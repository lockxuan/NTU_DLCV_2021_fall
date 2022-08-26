from dataset import p2_dataloader
from model import FCN32s, FCN8sVGG

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import argparse

import numpy as np
import math
import os
from PIL import Image

def pred_to_mask(pred, mask_path):
	cls_color = {
		0:  [0, 255, 255],
		1:  [255, 255, 0],
		2:  [255, 0, 255],
		3:  [0, 255, 0],
		4:  [0, 0, 255],
		5:  [255, 255, 255],
		6: [0, 0, 0],
	}
	mask = np.zeros((pred.shape[0], pred.shape[1], 3))
	for color, value in cls_color.items():
		mask[pred==color] = np.array(value)

	img = Image.fromarray(np.uint8(mask))
	img.save(mask_path)



def inference(model, test_loader, output_path, device):
	#output_path = os.path.join()
	#os.makedirs(output_path, exist_ok=True)
	output_path = os.path.join(output_path)

	model.eval()
	for step, batch in enumerate(test_loader):
		sat, paths = batch
		sat = sat.to(device)

		output = model(sat)

		preds = torch.max(output, dim=1)[1]
		for pred, path in zip(preds, paths):
			mask_path = os.path.join(output_path, path)
			pred_to_mask(pred.cpu(), mask_path)


		print('inference progress: {}/{}'.format(step+1, len(test_loader)), end='\r')
	print('', end='\r')










def main(args):
	transform = T.Compose([
			T.ToTensor(),
			T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
		])


	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = FCN8sVGG().to(device)
	test_loader = p2_dataloader(args.test_path, batch_size=8, train=False, shuffle=False, transform=transform)

	model.load_state_dict(torch.load(args.model_path, map_location=device))
	inference(model, test_loader, args.output_path, device)







if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--test_path", default='./hw1_data/p2_data/validation')
	parser.add_argument("--output_path", default='./p2_inference')
	parser.add_argument("--model_path", default='./p2_model.pth')
	args = parser.parse_args()
	main(args)



