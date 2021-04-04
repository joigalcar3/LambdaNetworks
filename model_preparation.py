from LabelSmoothing import LabelSmoothing
from resnet_lambda import resnet50_lambda
from resnet import resnet50

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime
import socket


def model_preparation(args, input_lst, model_name):
    '''
    Select the right model according to the model selected
    Args:
        args: user input parameters
        input_lst: first training batch
        model_name: name of the model chosen

    Returns: writer, resnet_nn, criterion, optimizer, label_smoothing, scheduler1, scheduler2, device, max_epoch, f
        writer: TensorBoard writer
        resnet_nn: model selected
        criterion: loss function
        optimizer: optimization function
        label_smoothing: label smoother
        scheduler1: first scheduler
        scheduler2: second scheduler
        device: deviced used for computation --> cpu or cuda
        max_epoch: starting epoch number
        f: logging file
    '''

    # Choose model file according to the chosen model type
    if args.model_type:
        resnet_nn = resnet50_lambda(pretrained=False, progress=True, num_classes=10, context_size=args.context_size,
                                    qk_size=args.qk_size, heads=args.heads, input_size=args.input_size,
                                    zero_init_residual=True)
    else:
        resnet_nn = resnet50(pretrained=False, progress=True, num_classes=10, zero_init_residual=True)

    # Print network architecture using torchsummary
    summary(resnet_nn, tuple(input_lst[0].shape), device='cpu')

    # Create a writer to write to Tensorboard
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', model_name, current_time + '_lr' + str(args.initial_lr) + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir)

    # Check if GPU available
    if torch.cuda.is_available():
        device = 'cuda'
        print('You have CUDA device.')
    else:
        device = 'cpu'
        print('Switch to GPU runtime to speed up computation.')

    # Bring model to device
    resnet_nn.to(device)

    # Define the criterion
    criterion = nn.CrossEntropyLoss()

    # Define the label smoother
    label_smoothing = LabelSmoothing(0.1)

    # Define the optimizer
    optimizer = optim.Adam(resnet_nn.parameters(), weight_decay=args.weight_decay,
                           betas=(0.9, 0.9999), lr=args.initial_lr)

    # Create linear scheduler for the first iterations
    lambda1 = lambda e: e+1
    scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # Create cosine scheduler
    steps = args.epochs - args.th       # This is Tmax according to the documentation of cosine annealing
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    # Restart from checkpoint
    max_epoch = 0
    if args.resume:
        folder_checkpoint = os.path.join(args.cp_dir, model_name)  # Location of the checkpoints
        if os.listdir(folder_checkpoint) != -1:
            # Maximum stored epoch
            max_epoch = max(list(map(lambda x: int(x[5:x.find('.')]), os.listdir(folder_checkpoint))))
            filepath = folder_checkpoint + '\\model' + str(max_epoch) + '.pt'  # Filepath to last checkpoint
            print("=> loading checkpoint '{}'".format(max_epoch))
            # Extract information from checkpoint
            checkpoint = torch.load(filepath, map_location=device)
            resnet_nn.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filepath, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(folder_checkpoint))

    # Create file for storing the logs
    f_path = ".\\logs\\" + model_name + "\\" +\
             current_time + '_lr' + str(args.initial_lr) + '_' + socket.gethostname() + ".txt"
    f = open(f_path, "a")

    return writer, resnet_nn, criterion, optimizer, label_smoothing, scheduler1, scheduler2, device, max_epoch, f