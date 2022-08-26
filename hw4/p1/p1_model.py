import torch
import torch.nn as nn







class Convnet(nn.Module):
	"""docstring for Convnet"""
	def __init__(self, in_channels=3, hid_channels=64, out_channels=64, in_mlp=1600, out_mlp=64):
		super(Convnet, self).__init__()
		self.encoder = nn.Sequential(
			conv_block(in_channels, hid_channels),
			conv_block(hid_channels, hid_channels),
			conv_block(hid_channels, hid_channels),
			conv_block(hid_channels, out_channels),
			)

		self.MLP = MLP(in_dim=in_mlp, out_dim=out_mlp)

		self.distance_nn = MLP(in_dim=out_mlp*2, out_dim=1)


	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.MLP(x)
		return x

	def cal_distance(self, a, b):
		n = a.size(0)
		m = b.size(0)
		
		a = a.unsqueeze(1).expand(n,m,-1).contiguous().view(n*m,-1)
		b = b.unsqueeze(0).expand(n,m,-1).contiguous().view(n*m,-1)

		x = torch.cat((a, b), dim=-1)
		x = self.distance_nn(x)

		return -x.view(n, m)


def conv_block(in_channels, out_channels):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, 3, padding=1),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(),
		nn.MaxPool2d(2)
		)



class MLP(nn.Module):
	"""docstring for MLP"""
	def __init__(self, in_dim, out_dim):
		super(MLP, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(in_dim, in_dim//4),
			nn.Dropout(0.5),
			nn.ReLU(),
			nn.Linear(in_dim//4, out_dim)
			)



	def forward(self, x):
		x = self.mlp(x)

		return x




if __name__ == '__main__':
	model = Convnet()

	#inp = torch.randn((4, 3, 84, 84))

	print(model)
		




