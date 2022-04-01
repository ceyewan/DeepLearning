# %%
"""
Import necessary libraries to create a generative adversarial network
The code is mainly developed using the PyTorch library
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
import PIL


# %%
"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Setting image parameters for the surface crack database
imsize = 227  # 227x227 images
in_chan = 3  # RGB
out_chan = 2  # 2 output classes: positive and negative

# Setting Hyperparameters
epochs = 100
learning_rate = 1e-3
decay = 1e-8
batch_size = 64
kernel_size = 5
# adding transforms to the image
transform = transforms.Compose([
    # cropping the centre
    transforms.CenterCrop(imsize),
    # adding random rotations
    transforms.RandomRotation([0, 360], resample=PIL.Image.BILINEAR),
    # transforming the dataset to Torch tensors
    transforms.ToTensor(),
    # normalising the image
    transforms.Normalize((0.0229,), (0.0957,))])
# loading the dataset and applying the transforms
dataset = datasets.ImageFolder(
    '../input/surface-crack-detection', transform=transform)

# %%
# Loading the data into a data loader, chop them into batches and shuffle the batches everytime this object is called
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True)
# Initialise a model
model = LeNet(in_chan=in_chan, out_chan=out_chan,
              imsize=imsize, kernel_size=kernel_size)
# Add Adam optimiser with a weight decay parameter to penalise big weights
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=decay)

epoch_trainaccs, epoch_valaccs = [], []
epoch_trainloss, epoch_valloss = [], []
for epoch in range(epochs):  # loop over the dataset multiple times
    # set the model to train mode (gradient acquiring)
    model.train()
    train_losses,  train_accs = [], []
    acc = 0
    # iterate through the train_loader
    for batch, (x_train, y_train) in enumerate(train_loader):
        # clear previous gradients
        model.zero_grad()
        # forward propagate the model
        pred = model(x_train)
        # Use the cross entropy loss because it's logistical regression
        loss = F.cross_entropy(pred, y_train)
        # backpropagate the loss
        loss.backward()
        # optimise the weights accordingly
        optimizer.step()
        # calculate the training accuracy
        acc = (pred.argmax(dim=-1) == y_train).to(torch.float32).mean()
        train_accs.append(acc.mean().item())
        train_losses.append(loss.item())
        print("Batch=", batch, " loss = ",
              loss.item(), " accuracy = ", acc.item())

    epoch_trainloss.append(np.mean(train_losses))
    epoch_trainaccs.append(np.mean(train_accs))
    print("Epoch = ", epoch, " Mean loss = ", np.mean(train_losses))

# %%
"""
Network Architectures
The following are the discriminator and generator architectures
"""


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 1)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return nn.Sigmoid()(x)


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 784)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 28, 28)
        return nn.Tanh()(x)


# %%
"""
Network training procedure
Every step both the loss for disciminator and generator is updated
Discriminator aims to classify reals and fakes
Generator aims to generate images as realistic as possible
"""
epochs = 100
for epoch in range(epochs):
    for idx, (imgs, _) in enumerate(train_loader):
        idx += 1

        # Training the discriminator
        # Real inputs are actual images of the MNIST dataset
        # Fake inputs are from the generator
        # Real inputs should be classified as 1 and fake as 0
        real_inputs = imgs.to(device)
        real_outputs = D(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)

        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)

        D_loss = loss(outputs, targets)
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Training the generator
        # For generator, goal is to make the discriminator believe everything is 1
        noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
        noise = noise.to(device)

        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
        G_loss = loss(fake_outputs, fake_targets)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        if idx % 100 == 0 or idx == len(train_loader):
            print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(
                epoch, idx, D_loss.item(), G_loss.item()))

    if (epoch+1) % 10 == 0:
        torch.save(G, 'Generator_epoch_{}.pth'.format(epoch))
        print('Model saved.')

# %%
