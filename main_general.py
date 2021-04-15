from nntrain import train
from nntest import test
from user_input import load_user_input
from data_preprocessing import data_preprocessing
from model_preparation import model_preparation
from log_data import log_data

import torch

import os
from tqdm import tqdm
import numpy as np
import time



if __name__ == "__main__":
    #%% Input parameters
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    args = load_user_input()

    # Map the network type to its name
    if args.model_type:
        model_name = 'Lambda'
    else:
        model_name = 'Baseline'

    # Check whether the checkpoint folder exists. If not, create it.
    folder_checkpoint = os.path.join(args.cp_dir, model_name)   # Path for storing the model info
    if not os.path.exists(folder_checkpoint):
        os.makedirs(folder_checkpoint)

    # Check whether the logs folder exists. If not, create it.
    logs_dir = ".\\logs\\" + model_name
    if not os.path.exists(logs_dir):   # Check whether the logs folder exists
        os.makedirs(logs_dir)

    #%% Define training and validation transforms
    train_loader, test_loader, input_lst = data_preprocessing(args)

    #%% Prepare the model
    writer, resnet_nn, criterion, optimizer, label_smoothing, scheduler1, scheduler2, device, max_epoch, f = \
        model_preparation(args, input_lst, model_name)

    #%% Train the resnet
    total_train_time = 0
    total_test_time = 0
    for epoch in tqdm(range(max_epoch, args.epochs)):  # loop over the dataset multiple times
        # Train on data
        start_train = time.time()
        train_loss, train_acc = train(train_loader, resnet_nn, optimizer, criterion,
                                      device, label_smoothing, args.smoothing)
        end_train = time.time()

        # Test on data
        start_test = time.time()
        test_loss, test_acc = test(test_loader, resnet_nn, criterion, device)
        end_test = time.time()

        # Print train and test accuracy and train and test loss. Log data. Save checkpoint. Store info for TensorBoard
        total_train_time, total_test_time = log_data(resnet_nn, start_train, train_loss, train_acc, end_train, total_train_time, start_test, test_loss,
                 test_acc, end_test, total_test_time, f, epoch, optimizer, folder_checkpoint, writer)

        # Obtain the new learning rate by choosing the right scheduler
        if epoch < args.th-1:
            scheduler1.step()
            print(scheduler1.get_last_lr())
        else:
            scheduler2.step()
            print(scheduler2.get_last_lr())

    print('Finished Training')

    # Write the information to the Tensorboard writer
    writer.flush()
    writer.close()
    f.close()


# Useful Tensorboard and Colab commands
# To be run on the terminal: tensorboard --logdir=runs
# To clean tensorboard: !rm -r runs
# !git clone https://github.com/joigalcar3/LambdaNetworks
# !python LambdaNetworks/main_general.py

# from google.colab import files
# !zip -r runs/Lambda/Apr02_11-13-58_21686eb8b94b.zip runs/Lambda/Apr02_11-13-58_21686eb8b94b
# files.download('./runs/Lambda/Apr02_11-13-58_21686eb8b94b.zip')
