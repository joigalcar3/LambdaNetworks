from nntrain import train
from nntest import test

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


import os
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import socket
from datetime import datetime

np.random.seed(0)
torch.manual_seed(0)

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        """
        Function of label smoothing according to the equations in page 7 of the paper: 'Rethinking the Inception
        Architecture for Computer Vision'
        """
        logprobs = nn.functional.log_softmax(x, dim=-1)     # Computes the log softmax of all the predictions. For each datapoint, there are 1000 numbers, each represents a label

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))   # Obtain the prediction obtained for the right labels. They should be as close to 1 as possible
        nll_loss = nll_loss.squeeze(1)       # Compute the first term of the equation of the paper, the predicted value of the labels
        smooth_loss = -logprobs.mean(dim=-1)   # Compute the second term, which is the average among all label predictions
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss    # Sum both terms weighted with the smoothing
        return loss.mean()    # Compute the batch loss

if __name__ == "__main__":
    #%% Input parameters
    b_size = 128               # Batch size: Table 4 of the original paper
    context_size = 23 * 23     # Context size: m
    qk_size = 16               # Key size: k
    heads = 4                  # Number of heads: h
    n_rows_plot = 8            # Number of rows to include in the plot of CIFAR-10
    n_col_plot = 8             # Number of columns to include in the plot of CIFAR-10
    epochs = 90              # Number of epochs: suggested by Robert-Jan Bruintjes
    weight_decay = 1e-4        # Weight decay for the Adam
    initial_lr = 0.01          # Initial learning rate
    th = 5                     # Threshold number of epochs to change scheduler
    model_type = 0            # Type of model: 0 = baseline; 1 = lambda
    resume = False             # Resume from the latest checkpoint
    smoothing = True           # switch which defines whether label smoothing should take place
    cp_dir = ".\Checkpoints"   # Base checkpoint folder

    if model_type:
        model_name = 'Lambda'
    else:
        model_name = 'Baseline'

    folder_checkpoint = os.path.join(cp_dir, model_name)   # Path for storing the model info
    if not os.path.exists(folder_checkpoint):
        os.makedirs(folder_checkpoint)

    #%% Define training and validation transforms
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor()])
    valid_transform = torchvision.transforms.ToTensor()

    # Preparing the dataset
    download_train = False if os.path.exists(".\CIFAR_10_train") == True else True
    download_test = False if os.path.exists(".\CIFAR_10_test") == True else True

    cifar10_train = torchvision.datasets.CIFAR10(root=".\CIFAR_10_train", train=True, download=download_train,
                                                 transform=train_transform)
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

    #%% Prepare the model
    if model_type:
        from resnet_lambda import *
        resnet_nn = resnet50(pretrained=False, progress=True, num_classes=10, context_size=context_size,
                             qk_size=qk_size,
                             heads=heads)
    else:
        from resnet import *
        resnet_nn = resnet50(pretrained=False, progress=True, num_classes=10)

    # Print network architecture using torchsummary
    summary(resnet_nn, tuple(input[0].shape), device='cpu')

    # Create a writer to write to Tensorboard
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', model_name, current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir)


    # Check if GPU available
    if torch.cuda.is_available():
        device = 'cuda'
        print('You have CUDA device.')
    else:
        device = 'cpu'
        print('Switch to GPU runtime to speed up computation.')

    # Bring model to device
    model = resnet_nn.to(device)

    # Define the criterion
    criterion = nn.CrossEntropyLoss()

    # Define the label smoother
    label_smoothing = LabelSmoothing(0.1)

    # Define the optimizer
    optimizer = optim.Adam(resnet_nn.parameters(), weight_decay=weight_decay, betas=(0.9, 0.9999), lr=initial_lr)

    # Create a scheduler
    lambda1 = lambda epoch: epoch+1
    scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    steps = epochs - th       # This is Tmax according to the documentation of cosine annealing
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    # Restart from checkpoint
    max_epoch = 0
    if resume:
        folder_checkpoint = os.path.join(cp_dir, model_name)
        if os.listdir(folder_checkpoint) != -1:
            max_epoch = max(list(map(lambda x: int(x[5:x.find('.')]), os.listdir(folder_checkpoint))))
            filepath = folder_checkpoint + '\model' + str(max_epoch) + '.pt'
            print("=> loading checkpoint '{}'".format(max_epoch))
            checkpoint = torch.load(filepath, map_location=device)
            start_epoch = checkpoint['epoch']
            resnet_nn.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filepath, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(folder_checkpoint))

    #%% Train the resnet
    for epoch in tqdm(range(max_epoch, epochs)):  # loop over the dataset multiple times
        # Train on data
        train_loss, train_acc = train(train_loader, resnet_nn, optimizer, criterion, device, label_smoothing, smoothing)

        # Test on data
        test_loss, test_acc = test(test_loader, resnet_nn, criterion, device)

        # Print train and test accuracy and train and test loss
        print("train_acc=", train_acc, "test_acc =", test_acc)
        print("train_loss=", round(train_loss.item(),2), "test_loss", round(test_loss.item(),2))

        # Obtain the new learning rate
        if epoch < th-1:
            scheduler1.step()
            print(scheduler1.get_last_lr())
        else:
            scheduler2.step()
            print(scheduler2.get_last_lr())

        # Save checkpoint
        if epoch % 5 == 0:
            path_save_epoch = folder_checkpoint + "\model" + str(epoch) + ".pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': resnet_nn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'train_loss': train_loss,
                'test_acc': test_acc,
                'train_acc': train_acc
            }, path_save_epoch)

        # Write metrics to Tensorboard
        writer.add_scalars("Loss", {'Train': train_loss, 'Test': test_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)

    print('Finished Training')
    writer.flush()
    writer.close()

# To be run on the terminal: tensorboard --logdir=runs
# To clean tensorboard: !rm -r runs