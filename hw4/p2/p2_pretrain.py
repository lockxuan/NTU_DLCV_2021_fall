import torch
#from p2_model import BYOL
from byol_pytorch import BYOL
from torchvision import models
from p2_dataset import p2_dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader



train_transform = T.Compose([
	T.Resize((128, 128)),
	T.RandomHorizontalFlip(p=20),
	T.ToTensor(),
	#T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

path = '../hw4_data/mini/train'

train_dataset = p2_dataset(path, train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50(pretrained=False).to(device)

learner = BYOL(
	resnet,
	image_size = 128,
	hidden_layer = 'avgpool'
)


opt = torch.optim.Adam(learner.parameters(), lr=3e-4)


for epoch in range(100):
	for i, batch in enumerate(train_loader):
		images, _ = batch
		images = images.to(device)
		loss = learner(images)

		opt.zero_grad()
		loss.backward()
		opt.step()
		learner.update_moving_average() # update moving average of target encoder

		print('[{}/{}] -> step: {}/{} | loss: {:.4f}'.format(epoch+1, 100, i+1, len(train_loader), loss), end='\r')

	if (epoch+1) % 10 == 0:
		torch.save(resnet.state_dict(), './ckpt/pretrained_resnet'+str(epoch+1)+'.pth')




