import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import glob
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    random.seed(888)
    torch.manual_seed(888)
    
    # load digit classifier
    net = Classifier()
    path = "./p2/Classifier.pth"
    load_checkpoint(path, net)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        net = net.to(device)

    #print(net)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5 )),
        ])

    class eval_dataset(Dataset):
        """docstring for eval_dataset"""
        def __init__(self, img_folder, transform=None):
            super(eval_dataset, self).__init__()
            
            self.img_files = glob.glob(os.path.join(img_folder, '*.png'))

            self.transform = transform

        def __getitem__(self, index):
            img = Image.open(self.img_files[index])


            if self.transform:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)

            label = int(self.img_files[index].split('/')[-1].split('_')[0])

            return img, label


        def __len__(self):
            return len(self.img_files)


    dataset = eval_dataset('./p2_inference', transform)

    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    total_correct = 0
    total_img = 0
    for step, batch in enumerate(loader):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = net(imgs)

        preds = torch.max(outputs, dim=1)[1]
        correct = (preds == labels).sum()
        total_correct += correct
        total_img += len(labels)

    print('acc: {}'.format(total_correct/total_img))








