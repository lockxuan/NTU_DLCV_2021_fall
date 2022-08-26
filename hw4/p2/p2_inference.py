import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import pandas as pd
import os
from PIL import Image
import numpy as np
from torchvision import models
import argparse
import csv

from p2_model import p2_model


pred_to_label = {13: 'Couch', 27: 'Helmet', 48: 'Refrigerator', 0: 'Alarm_Clock', 4: 'Bike', 5: 'Bottle',
				7: 'Calculator', 10: 'Chair', 36: 'Mouse', 34: 'Monitor', 59: 'Table', 42: 'Pen', 43: 'Pencil',
				22: 'Flowers', 52: 'Shelf', 32: 'Laptop', 56: 'Speaker', 54: 'Sneakers', 45: 'Printer', 8: 'Calendar',
				3: 'Bed', 30: 'Knives', 1: 'Backpack', 41: 'Paper_Clip', 9: 'Candles', 55: 'Soda', 11: 'Clipboards',
				24: 'Fork', 18: 'Exit_Sign', 31: 'Lamp_Shade', 63: 'Trash_Can', 12: 'Computer', 50: 'Scissors',
				64: 'Webcam', 53: 'Sink', 44: 'Postit_Notes', 25: 'Glasses', 20: 'File_Cabinet', 47: 'Radio',
				6: 'Bucket', 16: 'Drill', 15: 'Desk_Lamp', 62: 'Toys', 29: 'Keyboard', 38: 'Notebook', 49: 'Ruler',
				61: 'ToothBrush', 35: 'Mop', 21: 'Flipflops', 39: 'Oven', 58: 'TV', 17: 'Eraser', 60: 'Telephone',
				28: 'Kettle', 14: 'Curtains', 37: 'Mug', 19: 'Fan', 46: 'Push_Pin', 2: 'Batteries', 40: 'Pan',
				33: 'Marker', 57: 'Spoon', 51: 'Screwdriver', 26: 'Hammer', 23: 'Folder'}


class p2_dataset(Dataset):
	"""docstring for p2_dataset"""
	def __init__(self, csv_path, path, transform):
		super(p2_dataset, self).__init__()

		df = pd.read_csv(csv_path)

		self.imgs_path = [os.path.join(path, file_name) for file_name in df['filename']]
		self.file_names = [file_name for file_name in df['filename']]

		self.transform = transform

	def __getitem__(self, index):

		img = self.transform(Image.open(self.imgs_path[index]))

		return img, self.file_names[index]

	def __len__(self):
		return len(self.imgs_path)



def main(args):
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	feature_extractor = models.resnet50(pretrained=False)
	feature_extractor.load_state_dict(torch.load(args.extractor, map_location='cpu'))
	feature_extractor = feature_extractor.to(device)

	classifier = p2_model()
	classifier.load_state_dict(torch.load(args.classifier, map_location='cpu'))
	classifier = classifier.to(device)

	transform = T.Compose([
			T.Resize((128, 128)),
			T.ToTensor(),
			T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])

	val_dataset = p2_dataset(args.val_csv, args.val_path, transform)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

	feature_extractor.eval()
	classifier.eval()

	with open(args.output_csv, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['id', 'filename', 'label'])
		_id = 0
		for step, batch in enumerate(val_loader):
			imgs, file_names = batch
			imgs = imgs.to(device)

			features = feature_extractor(imgs)
			outputs = classifier(features)

			preds = torch.max(outputs, dim=1)[1]

			for n, p in zip(file_names, preds):
				writer.writerow([_id, n, pred_to_label[p.item()]])
				_id += 1

			print('step: {}/{}'.format(step+1, len(val_loader)), end='\r')





def argument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--val_csv', default='../hw4_data/office/val.csv', type=str)
	parser.add_argument('--val_path', default='../hw4_data/office/val', type=str)
	parser.add_argument('--output_csv', default='./inf_p2.csv', type=str)
	parser.add_argument('--extractor', default='./p2_extractor_c.pth', type=str)
	parser.add_argument('--classifier', default='./p2_classifier_c.pth', type=str)
	parser.add_argument('--device', default='cuda:0', type=str)

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = argument()
	main(args)

