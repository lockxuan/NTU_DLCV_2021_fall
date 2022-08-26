from p1_model import DCGAN_G
import torch
import argparse
import random
import os
from torchvision.utils import save_image
import torchvision.transforms as T



def fix_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)


def inference(model_G, save_dir, device, num_photos=1000, hidden_dim=100):
	invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
										std = [ 1/0.5, 1/0.5, 1/0.5 ]),
							T.Normalize(mean = [ -0.5, -0.5, -0.5 ],
										std = [ 1., 1., 1. ]),
						])

	noise = torch.randn(num_photos, hidden_dim, 1, 1).to(device)

	outputs = model_G(noise)
	#save_image(outputs.data[:32], "p1/p1_2.png", normalize=True)

	os.makedirs(save_dir, exist_ok=True)
	for i, img in enumerate(outputs):
		save_path = os.path.join(save_dir, '{0:04d}.png'.format(i))
		save_image(invTrans(img), save_path)






def main(args):
	fix_seed(args.seed)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model_G = DCGAN_G(100)
	model_G.load_state_dict(torch.load(args.model_path, map_location=device))
	model_G = model_G.to(device)
	model_G.eval()

	inference(model_G, args.save_dir, device)






if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_dir", default='./inference')
	parser.add_argument("--model_path", default='./p1_G.pth')
	parser.add_argument("--seed", default=0)
	args = parser.parse_args()
	main(args)



