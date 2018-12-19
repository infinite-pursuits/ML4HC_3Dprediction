import argparse
from mnist import MNIST
import numpy as np
import read_qmnist
import os
import torch
from network2 import Network
import torch.utils.data as Data
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset import MyDataset
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))*float(0.5)
        m.weight.data.normal_(0.0, variance)


def find_mean_image(data_loader):
    for batch_no, (images, _) in enumerate(data_loader):
        a = torch.mean(images, 0)
    return a


def train(model, loss_criterion, optimizer, data_loader, test_data_loader, epochs, scheduler, momentum, data_train, batchsize, test_size):
    hist = []
    print("In train ")
    for epoch in range(0, epochs):
        model.train()
        for batch_no, (images, target_labels) in enumerate(data_loader):
            
            optimizer.zero_grad()
            vals= model(images)
            loss = loss_criterion(vals, target_labels)
            loss.backward()
            optimizer.step()
            """if batch_no % 10 == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.4f}'.format(
                    epoch, batch_no * len(images), len(data_loader.dataset), loss.item()), flush=True)"""
        """for param_group in optimizer.param_groups:
            print(param_group['lr'])
            print(param_group['momentum'])

            if epoch == epochs//4 and momentum:
                param_group['momentum'] = param_group['momentum']/float(2)"""
        scheduler.step()
        #if epoch % 10==0:
        #    data_loader = Data.DataLoader(data_train, batch_size=batchsize*2, shuffle=True, num_workers=1) 
        train_loss, train_correct = quick_test(model, data_loader, loss_criterion, 256)
        train_acc = float(train_correct) * 100.0 / 60000.0
        test_loss, test_correct = quick_test(model, test_data_loader, loss_criterion, 256)
        acc = float(test_correct) * 100.0 / float(test_size)
        hist.append(acc)
        print("Epoch: {}".format(epoch))
        print("Training Accuracy: {:.4f}, Training Loss: {:.4f}".format(train_acc, train_loss))
        print("Testing Accuracy: {:.4f}, Testing Loss: {:.4f}".format(acc, test_loss))


def quick_test(model, data_loader, loss_criterion, batch_size):
    loss = 0.0
    correct = 0
    model.eval()

    for batch_no, (images, labels) in enumerate(data_loader):
        vals = model(images)
        loss += loss_criterion(vals, labels)
        log_soft = torch.nn.LogSoftmax()
        log_soft_vals = log_soft(vals)
        pred_labels = log_soft_vals.data.max(1)[1]
        correct += (pred_labels == labels).sum()
    return loss, correct


def test(model, data_loader, filepath, loss_criterion, batch_size):
    correct = 0
    file = filepath+'/predictedvsdesired.txt'
    loss = 0.0
    full_dump = np.array([0, 0, 0, 0])
    model.eval()

    for batch_no, (images, labels) in enumerate(data_loader):
        vals = model(images)
        loss += loss_criterion(vals, labels)
        log_soft = torch.nn.LogSoftmax()
        log_soft_vals = log_soft(vals)
        pred_labels = log_soft_vals.data.max(1)[1]
        correct += (pred_labels == labels).sum()
        diff = (pred_labels == labels)
        diff = (diff == 0)*1
        serial = np.array((range(batch_no * batch_size, (batch_no * batch_size)+len(diff))))
        serial = serial.reshape(len(serial), -1)
        diff = diff.reshape(len(diff), -1)
        test_l = labels.reshape(len(labels), -1)
        pred = pred_labels.reshape(len(pred_labels), -1)
        dump = np.hstack((serial, diff, pred, test_l))
        full_dump = np.vstack((full_dump, dump))
    np.savetxt(file, full_dump, fmt='%d', delimiter=' ')
    return loss, correct


def run(config):
    cwd = os.getcwd()
    #os.chdir(".."+'/datasets')
    datadir = os.getcwd()
    print(cwd, flush=True)
    print(datadir, flush=True)

    if config.tt == 0:

        mndata = MNIST("mnist-samples")
        train_images, train_labels = mndata.load_training()
        test_images, test_labels = mndata.load_testing()
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

    elif config.tt == 1:

        train_images_path = os.getcwd()+'/qmnist-v2-dataset/qmnist-train-images-idx3-ubyte'
        train_images = read_qmnist.read_image_file(train_images_path, return_np=True)
        train_labels_path = os.getcwd()+'/qmnist-v2-dataset/qmnist-train-labels-tsv'
        train_labels = read_qmnist.read_label_file_from_tsv(train_labels_path, usecols=0)
        test_images_path = os.getcwd()+'/qmnist-v2-dataset/qmnist-test-images-idx3-ubyte'
        test_images = read_qmnist.read_image_file(test_images_path, return_np=True)
        test_labels_path = os.getcwd()+'/qmnist-v2-dataset/qmnist-test-labels-tsv'
        test_labels = read_qmnist.read_label_file_from_tsv(test_labels_path, usecols=0)
        train_images = train_images.reshape(len(train_images), -1)
        test_images = test_images.reshape(len(train_images), -1)

    elif config.tt == 2:

        mndata = MNIST("mnist-samples")
        train_images, train_labels = mndata.load_training()
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images_path = os.getcwd() + '/qmnist/qmnist-test-images-idx3-ubyte'
        test_images = read_qmnist.read_image_file(test_images_path, return_np=True)
        test_labels_path = os.getcwd() + '/qmnist/qmnist-test-labels-tsv'
        test_labels = read_qmnist.read_label_file_from_tsv(test_labels_path, usecols=(0))
        test_images = test_images.reshape(len(train_images), -1)

    elif config.tt == 3:

        train_images_path = os.getcwd() + '/qmnist-v2-dataset/qmnist-train-images-idx3-ubyte'
        train_images = read_qmnist.read_image_file(train_images_path, return_np=True)
        train_labels_path = os.getcwd() + '/qmnist-v2-dataset/qmnist-train-labels-tsv'
        train_labels = read_qmnist.read_label_file_from_tsv(train_labels_path, usecols=(0))
        train_images = train_images.reshape(len(train_images), -1)
        mndata = MNIST("mnist-samples")
        test_images, test_labels = mndata.load_testing()
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

    print(config, flush=True)
    size_test = len(test_labels)
    if config.testingsetpart != 1.0:
        no_of_testing_samples = int(config.testingsetpart * size_test)
        if config.testingsetend == -1:
            test_images = test_images[0:no_of_testing_samples]
            test_labels = test_labels[0:no_of_testing_samples]
        elif config.testingsetend == 0:
            mid = size_test//2
            start = mid-1-((no_of_testing_samples//2)-1)
            end = mid+(no_of_testing_samples//2)
            test_images = test_images[start: end]
            test_labels = test_labels[start: end]
        elif config.testingsetend == 1:
            test_images = test_images[(size_test-no_of_testing_samples):size_test]
            test_labels = test_labels[(size_test-no_of_testing_samples):size_test]
    print("Training Images size: {:d} {:d}, \n"
          "Training Labels size: {:d}, \n"
          "Testing Images size: {:d} {:d}, \n"
          "Testing Labels size: {:d}".format(train_images.shape[0], train_images.shape[1], len(train_labels),
                                             test_images.shape[0], test_images.shape[1], len(test_labels)), flush=True)

    train_images = torch.from_numpy(train_images).float()*0.01
    train_images = train_images.to(device)
    train_labels = torch.from_numpy(train_labels).long().to(device)
    test_images = torch.from_numpy(test_images).float()*0.01
    test_images = test_images.to(device)
    test_labels = torch.from_numpy(test_labels).long().to(device)

    os.chdir(cwd)
    print(os.getcwd(), flush=True)
    filepath = os.getcwd() + '/' + config.expname
    os.makedirs(filepath, exist_ok=True)
    lr = config.lr
    epochs = config.epochs
    data_train = MyDataset(60000, train_images, train_labels)
    if config.tt == 0 or config.tt == 3:
        size = 10000
    else:
        size = 60000

    model = Network(28*28, config.hidden_units1, 10, config.seed, config.old_init)
    if config.gaussian_init:
        model.apply(weight_init)
    model.float().to(device)
    data_mean_loader = Data.DataLoader(data_train, batch_size=60000)
    mean_image = find_mean_image(data_mean_loader)
    train_images = train_images - mean_image
    test_images = test_images - mean_image
    data_train = MyDataset(60000, train_images, train_labels)
    data_test = MyDataset(size, test_images, test_labels)
    """plt.imshow(mean_image.detach().float().numpy().reshape((28, 28)))
    plt.show()
    plt.imshow(train_images[0].detach().float().numpy().reshape((28,28)))
    plt.show()
    train_images = train_images-mean_image
    plt.imshow(train_images[0].detach().float().numpy().reshape((28,28)))
    plt.show()
    train_images = train_images+mean_image
    plt.imshow(train_images[0].detach().float().numpy().reshape((28,28)))
    plt.show()"""

    data_train_loader = Data.DataLoader(data_train, batch_size=config.batchsize, shuffle=True, num_workers=1)
    data_test_loader = Data.DataLoader(data_test, batch_size=config.batchsize, num_workers=1)
    if config.m:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config.mval, weight_decay = 0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = 0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, config.scheduler_step, gamma=config.gamma)
    cel_criterion = torch.nn.CrossEntropyLoss()

    train(model, cel_criterion, optimizer, data_train_loader, data_test_loader, epochs, scheduler, config.m, data_train, config.batchsize, size)
    test_loss, correct = test(model, data_test_loader, filepath, cel_criterion, config.batchsize)
    acc = float(correct) * 100.0 / float(len(test_labels))
    avg_test_loss = test_loss / len(test_labels)
    print("Accuracy: {:.4f}".format(acc), flush=True)
    print("Error: {:.5f}".format(100-acc), flush=True)
    print("Testing Loss : {:.5f}".format(avg_test_loss), flush=True)
    resultfile = filepath + '/resultfile.txt'
    file = open(resultfile, 'w')
    file.write("Configs\n")
    file.write("Exp name: %s \nHidden Units1: %s \nHidden Units2: %s \nLearning Rate: %s \n"
               "BatchSize: %s \nEpochs: %s \nSeed: %s \nInit: %s \n"
               % (str(config.expname), str(config.hidden_units1), str(config.hidden_units2), str(config.lr),
            str(config.batchsize), str(config.epochs), str(config.seed), str(config.old_init)))
    file.write("Accuracy: ")
    file.write(str(acc))
    file.write("\nTest Loss: ")
    file.write(str(avg_test_loss))
    file.write('\n')
    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expname", help="Name of the experiment")
    parser.add_argument("--tt", default=1, type=int, help="0-TMTM, 1-TQTQ, 2-TMTQ, 3-TQTM ")
    parser.add_argument("--testingsetpart", default=1.0, type=float, help="How much of the testing set should be used. 0-1")
    parser.add_argument("--testingsetend", default=-1, type=int, help="Which end do you want to take the testing set part. -1 - left, 0-mid, 1-right")
    parser.add_argument("--crossval", action='store_true', help="Do you want to cross validate or not. Default = False")
    parser.add_argument("--hidden_units1", default=800, type=int, help="Hidden units for 1st hidden layer")
    parser.add_argument("--hidden_units2", default=150, type=int, help="Hidden units for 2nd hidden layer")
    parser.add_argument("--lr", default=0.001, type=float, help="Input Learning Rate")
    parser.add_argument("--batchsize", default=32, type=int, help="Batch size for training")
    parser.add_argument("--epochs", default=210, type=int, help="Number of epochs")
    parser.add_argument("--batchnorm", action='store_true', help="Whether or not to do Batch Normalization?")
    parser.add_argument("--seed", default=1000, type=int, help="Random Seed")
    parser.add_argument("--old_init", action='store_true', help="Whether to initialize using your old formula?")
    parser.add_argument("--gaussian_init", action='store_true', help="Whether to do gaussian initialization?")
    parser.add_argument("--scheduler_step", default=30, type=int, help="Step LR after how many steps?")
    parser.add_argument("--gamma", default=0.316, type=float, help="Scheduler gamma")
    parser.add_argument("--m", action='store_true', help="Momentum???")
    parser.add_argument("--mval", default=0.9, type=float, help="momentum value")
    config = parser.parse_args()
    run(config)

