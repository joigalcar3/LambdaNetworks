from resnet import *

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import os
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def train(train_loader, net, optimizer, criterion):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return avg_loss / len(train_loader), 100 * correct / total


def test(test_loader, net, criterion):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        net: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        for data in test_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # keep track of loss and accuracy
            avg_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return avg_loss / len(test_loader), 100 * correct / total

if __name__ == "__main__":
    #%% Input parameters
    b_size = 128   # Batchsize: Table 4 of the original paper
    n_rows_plot = 8    # Number of rows to include in the plot of CIFAR-10
    n_col_plot = 8    # Number of columns to include in the plot of CIFAR-10
    epochs = 100     # Number of epochs: suggested by Robert-Jan Bruintjes

    #%% define transforms
    valid_transform = torchvision.transforms.ToTensor()

    # Preparing the dataset
    download_train = True if os.path.exists(".\CIFAR_10_train") == True else False
    download_test = True if os.path.exists(".\CIFAR_10_test") == True else False

    cifar10_train = torchvision.datasets.CIFAR10(root=".\CIFAR_10_train", train=True, download=download_train,
                                                 transform=valid_transform)
    cifar10_test = torchvision.datasets.CIFAR10(root=".\CIFAR_10_test", train=False, download=download_test,
                                                transform=valid_transform)

    train_loader = DataLoader(cifar10_train, batch_size=b_size, shuffle=True)
    test_loader = DataLoader(cifar10_test, batch_size=b_size, shuffle=True)

    input, label = next(iter(train_loader))

    # Plot a random group of images from the training set
    plt.figure()
    for i in range(n_rows_plot*n_col_plot):
        plt.subplot(n_rows_plot, n_col_plot, i+1)
        plt.imshow(input[i].permute(1, 2, 0))
    plt.show()

    #%% Train the model
    resnet_nn = resnet50(pretrained=False, progress=True)

    # Print network architecture using torchsummary
    summary(resnet_nn, tuple(input[0].shape), device='cpu')

    # Create a writer to write to Tensorboard
    writer = SummaryWriter()

    # Define the criterion
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(resnet_nn.parameters())

    # Train the resnet
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        # Train on data
        train_loss, train_acc = train(train_loader, resnet_nn, optimizer, criterion)

        # Test on data
        test_loss, test_acc = test(test_loader, resnet_nn, criterion)

        # Write metrics to Tensorboard
        writer.add_scalars("Loss", {'Train': train_loss, 'Test': test_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)

    print('Finished Training')
    writer.flush()
    writer.close()
