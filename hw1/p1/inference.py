import torch
import torch.nn as nn
import os, glob
import numpy as np
import torchvision.transforms as T
import argparse

from dataset import p1_dataloader
from model import p1_model

import numpy as np
import csv



def load_model(model, modelPath, device):
	model.load_state_dict(torch.load(modelPath, map_location=device))
	return model

def inference(model, test_loader, device, model_path, output_csv):
	model = load_model(model, model_path, device)
	model.eval()


	with torch.no_grad():
		with open(output_csv, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['image_id', 'label'])

			for step, batch in enumerate(test_loader):
				imgs, labels = batch
				imgs = imgs.to(device)

				output = model(imgs)

				_, pred = torch.max(output, 1)
				for file_name, out in zip(labels, pred.cpu().detach().numpy()):
					writer.writerow([file_name, out])


				print('testing progress: {}/{}'.format(step+1, len(test_loader)), end='\r')





def main(args):
	
	transform = T.Compose([
		T.Resize(224),
		T.ToTensor(),
		T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
	])

	
	test_loader = p1_dataloader(args.test_path, batch=32, train=False, shuffle=False, transform=transform)


	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = p1_model().to(device)

	inference(model, test_loader, device, model_path=args.model_path, output_csv=args.output_csv)






if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--test_path", default='../hw1_data/p1_data/val_50')
	parser.add_argument("--output_csv", default="./p1_output.csv")
	parser.add_argument("--model_path", default="./p1_model.pth")
	args = parser.parse_args()
	main(args)


