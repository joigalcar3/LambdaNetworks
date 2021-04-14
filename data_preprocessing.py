import torchvision
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

def data_preprocessing(args):
    '''
    Preprocessing the data and creating the dataloaders for training and testing
    Args:
        args: user input

    Returns:
        train_loader: iterator containing training data
        test_loader: iterator containing test data
        input_lst: first training data batch

    '''
    # Define training, testing and plotting transformations
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    plot_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor()])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # Preparing the dataset
    download_train = False if os.path.exists(".\\CIFAR_10_train") else True
    download_test = False if os.path.exists(".\\CIFAR_10_test") else True

    cifar10_train = torchvision.datasets.CIFAR10(root=".\\CIFAR_10_train", train=True, download=download_train,
                                                 transform=train_transform)
    cifar10_plot = torchvision.datasets.CIFAR10(root=".\\CIFAR_10_train", train=True, download=download_train,
                                                 transform=plot_transform)
    cifar10_test = torchvision.datasets.CIFAR10(root=".\\CIFAR_10_test", train=False, download=download_test,
                                                transform=test_transform)

    # Create train and test data loaders, iterators
    train_loader = DataLoader(cifar10_train, batch_size=args.b_size, shuffle=True)
    plot_loader = DataLoader(cifar10_plot, batch_size=args.b_size, shuffle=True)
    test_loader = DataLoader(cifar10_test, batch_size=args.b_size, shuffle=True)

    # Obtain one data training batch
    input_lst, label = next(iter(plot_loader))

    # Plot a random group of images from the training set
    plt.figure()
    for i in range(args.n_rows_plot*args.n_col_plot):
        plt.subplot(args.n_rows_plot, args.n_col_plot, i+1)
        plt.imshow(input_lst[i].permute(1, 2, 0))
    plt.show()

    return train_loader, test_loader, input_lst
