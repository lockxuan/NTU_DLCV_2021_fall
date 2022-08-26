from p2_model import ACGAN_G
import torch
import argparse
import random
import os
from torchvision.utils import save_image
import torchvision.transforms as T
import numpy as np



def fix_seed(seed):
	print('fix seed:', seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)


def inference(model_G, save_dir, device, num_photos=1000, hidden_dim=110):
	invTrans = T.Compose([ T.Resize(28),
							T.Normalize(mean = [ 0., 0., 0. ],
										std = [ 1/0.5, 1/0.5, 1/0.5 ]),
							T.Normalize(mean = [ -0.5, -0.5, -0.5 ],
										std = [ 1., 1., 1. ]),
						])

	### make noise
	fixed_noise = torch.randn(1000, hidden_dim, 1, 1)
	fake_noise = np.random.normal(0, 1, (1000, hidden_dim))
	fake_onehot = np.zeros((1000, 10))
	for i in range(10):
		fake_classes = np.ones(100, dtype=int)*i
		fake_onehot[np.arange(i*100, (i+1)*100), fake_classes] = 1
		fake_noise[np.arange(i*100, (i+1)*100), :10] = fake_onehot[np.arange(i*100, (i+1)*100)]
	fake_noise = (torch.from_numpy(fake_noise))
	fixed_noise.data.copy_(fake_noise.view(1000, hidden_dim, 1, 1))
	fixed_noise = fixed_noise.to(device)

	
	outputs = model_G(fixed_noise)
	outputs = invTrans(outputs)
	

	os.makedirs(save_dir, exist_ok=True)
	for i in range(10):
		for j, img in enumerate(outputs[i*100:(i+1)*100]):
			save_path = os.path.join(save_dir, str(i)+'_{0:03d}.png'.format(j+1))
			save_image(img, save_path)
	
	"""
	### save smaple image 0-9 (per num. 10 imgs)
	out_sample = np.arange(10)
	for j in range(1,10):
		out_sample = np.concatenate((out_sample, np.arange(j*100, j*100+10)))
	save_image(outputs.data[out_sample], "p2/p2_2.png", nrow=10)
	"""




def main(args):
	fix_seed(args.seed)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model_G = ACGAN_G(110)
	model_G.load_state_dict(torch.load(args.model_path, map_location=device))
	model_G = model_G.to(device)
	model_G.eval()

	inference(model_G, args.save_dir, device)






if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_dir", default='./p2_inference')
	parser.add_argument("--model_path", default='./ckpt/p2_G.pth')
	parser.add_argument("--seed", type=int, default=2022)
	args = parser.parse_args()
	main(args)



