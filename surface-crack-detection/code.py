import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms, models, utils
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

batch_size = 64
learning_rate = 0.01
num_epoches = 20

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageFolder('../input/surface-crack-detection', transform=transform)
# full_ds = dataset
# # train_size = int(0.8 * len(full_ds))
# # validate_size = len(full_ds) - train_size
# train_ds = torch.utils.data.random_split(
#     full_ds, train_size)
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = CNN()

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

epoch_list = []
acc_list = []
epoch = 0
for data in train_loader:
    img, label = data
    img = Variable(img)
    if (torch.cuda.is_available()):
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    out = model(img)
    loss = criterion(out, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if epoch % 10 == 0:
        epoch_list.append(epoch)
        acc_list.append(loss.item())
        print('*' * 10)
        print('epoch{}'.format(epoch))
        print('loss is {:.4f}'.format(loss.item()))


torch.save(model, 'model.pt')

plt.plot(epoch_list, acc_list)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
# model.eval()
# eval_loss = 0
# eval_acc = 0

# for data in test_loader:
#     img, label = data
#     img = Variable(img)
#     if (torch.cuda.is_available()):
#         img = Variable(img).cuda()
#         label = Variable(label).cuda()
#     else:
#         img = Variable(img)
#         label = Variable(label)
#     out = model(img)
#     loss = criterion(out, label)
#     eval_loss += loss.item() * label.size(0)
#     _, pred = torch.max(out, 1)
#     num_correct = (pred == label).sum()
#     eval_acc += num_correct.item()
# print('Test Loss:{:.6f}, Acc:{:.6f}'.format(
#     eval_loss / (len(test_ds)), eval_acc / (len(test_ds))))
