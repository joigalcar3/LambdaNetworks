import torch


def log_data(resnet_nn, start_train, train_loss, train_acc, end_train, total_train_time, start_test, test_loss, test_acc,
             end_test, total_test_time, f, epoch, optimizer, folder_checkpoint, writer):
    '''
    Data logger and printer
    Args:
        resnet_nn: selected model
        start_train: time at which the current epoch started training
        train_loss: training loss
        train_acc: training accuracy
        end_train: time at which the current epoch stopped training
        total_train_time: total time spent training
        start_test: time at which the current epoch started testing
        test_loss: testing loss
        test_acc: testing accuracy
        end_test: time at which the current epoch stopped testing
        total_test_time: total time spent testing
        f: logging file
        epoch: current epoch
        optimizer: chosen optimizer
        folder_checkpoint: folder location of the checkpoints
        writer: writer to Tensorboard

    Returns:

    '''
    # Obtain training time
    elapsed_train = end_train-start_train
    total_train_time += elapsed_train

    # Obtain testing time
    elapsed_test = end_test - start_test
    total_test_time += elapsed_test

    # Print train and test accuracy and train and test loss
    print("train_acc = ", train_acc, "test_acc = ", test_acc)
    print("train_loss = ", round(train_loss.item(), 2), "test_loss = ", round(test_loss.item(), 2))

    # Store information in logs
    f.write("epoch = " + str(epoch) + "\n")
    f.write("\tlearning rate = " + str(optimizer.param_groups[0]['lr']) + "\n")
    f.write("\ttrain acc = " + str(train_acc) + " --- test acc = " + str(test_acc) + "\n")
    f.write("\ttrain epoch time = " + str(elapsed_train) + " s\n")
    f.write("\ttrain loss = " + str(round(train_loss.item(), 2)) + " --- test loss = " + str(round(test_loss.item(),
                                                                                                   2)) + "\n")
    f.write("\ttest epoch time = " + str(elapsed_test) + " s\n")
    f.write("\ttotal time train = " + str(total_train_time) + " s\n")
    f.write("\ttotal time test = " + str(total_test_time) + " s\n")

    # Save checkpoint
    if epoch % 5 == 0:
        path_save_epoch = folder_checkpoint + "\\model" + str(epoch) + ".pt"
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