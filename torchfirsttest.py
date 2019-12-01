import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import torch.nn.functional as Functional_def


BATCH_SIZE = 128
NUM_EPOCHS = 10
learn_rate=0.8

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 625)
        self.fc2 = nn.Linear(625, 10)

    def forward(self, x):
        x = self.pool1(Functional_def.relu(self.conv1(x)))
        x = self.pool2(Functional_def.relu(self.conv2(x)))
        x = self.pool3(Functional_def.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = Functional_def.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    # TODO:define model


model = SimpleNet()

# TODO:define loss function and optimiter
criterion =nn.NLLLoss()
optimizer =torch.optim.SGD(model.parameters(),lr=learn_rate)


# train and evaluate
for epoch in range(NUM_EPOCHS):
    for images, labels in tqdm(train_loader):
        images = images.to('cpu')
        labels = labels.to('cpu')
        outputs = Functional_def.softmax(model(images))
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to()
        labels = labels.to()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the  test images: {} %'.format(100 * correct / total))
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to()
        labels = labels.to()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the train images: {} %'.format(100 * correct / total))