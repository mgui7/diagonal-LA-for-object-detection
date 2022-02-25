# Standard imports
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)  # bs x 1 x 28 x 28 -> bs x 5 x 24 x 24
        self.pool = nn.MaxPool2d(2, 2)  # bs x 5 x 24 x 24 -> bs x 5 x 12 x 12
        self.conv2 = nn.Conv2d(5, 10, 5)  # bs x 10 x 8 x 8
        self.fc1 = nn.Linear(10 * 4 * 4, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch: bs x 160
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaseNet(object):
    def __init__(self, lr=1e-3, epoch=3, batch_size=32, device='cpu'):
        print('Creating Net!!')
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
        self.create_net()
        self.create_opt()

    def create_net(self):
        torch.manual_seed(42)
        if self.device == 'cuda':
            torch.cuda.manual_seed(42)
        self.model = Net()
        if self.device == 'cuda':
            self.model.cuda()
        print('Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def save(self, filename):
        print('Writting %s\n' % filename)
        torch.save({
            'epoch': self.epoch,
            'lr': self.lr,
            'model': self.model,
            'optimizer': self.optimizer}, filename)

    def train(self, data, criterion):
        self.model.train()
        for epoch in range(self.epoch):
            for images, labels in tqdm(data):
                logits = self.model(images.to(self.device))
                loss = criterion(logits, labels.to(self.device))
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

    def eval(self, data):
        self.model.eval()
        logits = torch.Tensor().to(self.device)
        targets = torch.LongTensor()
        with torch.no_grad():
            for images, labels in tqdm(data):
                logits = torch.cat([logits, self.model(images.to(self.device))])
                targets = torch.cat([targets, labels])
        return torch.nn.functional.softmax(logits, dim=1), targets

    def accuracy(self, predictions, labels):
        print(f"Accuracy: {100 * np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.numpy()):.2f}%")