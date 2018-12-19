import argparse
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from dataset3D import BrainImages
import models3d
import torch.optim as optim
import torch.nn.init as I
import os
import numpy as np
import torch
import torch.nn as nn
import statistics as stat
from tensorboardX import SummaryWriter
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(1000)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        I.xavier_normal_(m.weight.data)
    if classname.find('Linear') != -1:
        I.normal_(m.weight.data)
        m.bias.data.fill_(0)

def pickling(file,path):
    pickle.dump(file,open(path,'wb'))

def unpickling(path):
    file_return=pickle.load(open(path,'rb'))
    return file_return

def test(f, model, dataloader, loss_criterion_list, paramstopredict, means, stdev, target_prep, check_val=20):
    with torch.no_grad():
        model.train(False)
        loss_list = [0] * paramstopredict
        correct_list = [0] * paramstopredict
        for batch_no, data in enumerate(dataloader['test']):
            torch.cuda.empty_cache()
            x = data['x'].to(device)
            y = data['y'].to(device)
            predicted = model(x)
            for i in range(paramstopredict):
                loss_list[i] += loss_criterion_list[i](predicted[:, i], y[:, i])
                if target_prep:
                    predicted[:, i] = predicted[:, i] * stdev[i] + means[i]
                    y[:, i] = y[:, i] * stdev[i] + means[i]
            print("Test : Predicted Desired {} {}".format(predicted, y), flush=True)
            f.write("Test : Predicted Desired {} {}\n".format(predicted, y))
            for i in range(paramstopredict):
                a = np.abs(np.around(np.array(predicted[:,i].detach().cpu()), decimals=3))
                b = np.abs(np.around(np.array(y[:, i].detach().cpu()), decimals=3))
                correct_list[i] += (np.abs(a - b) <= check_val).sum()
    return loss_list, correct_list


def train(lr, paramlist, target_norm, logger, inclr, real_batchsize, sch, scheduler, clip, f, model, optimizer, loss_criterion_list, dataloader, data_sizes, num_epochs, folderpath, paramstopredict,
          means, stdev, verbose=False):
    best_loss = np.full(paramstopredict, np.inf).tolist()
    loss_hist = {'train': [], 'validate': []}
    orig_clip = clip
    orig_lr = lr
    changed_lr = 0
    for epoch_num in range(num_epochs):
        torch.cuda.empty_cache()
        for phase in ['train', 'validate']:
            running_loss = [0] * paramstopredict
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            total_loss = torch.zeros([1, paramstopredict], dtype=torch.float32)
            for batch_idx, data in enumerate(dataloader[phase]):
                loss_list = []
                optimizer.zero_grad()
                x = data['x'].to(device)
                y = data['y'].to(device)
                output = model(x)

                for i in range(paramstopredict):
                    loss_list.append(loss_criterion_list[i](output[:, i], y[:, i]))
                # https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/7
                loss = sum(loss_list)
                torch.nn.utils.clip_grad_value_(model.parameters(), clip)

                if phase == 'train':
                    if real_batchsize == 1:
                        loss.backward()
                        optimizer.step()
                    if real_batchsize>1:
                        total_loss = total_loss + torch.FloatTensor(loss_list)
                        if batch_idx!= 0 and (batch_idx % real_batchsize == 0 or batch_idx == 667):
                            ave_loss = total_loss/real_batchsize
                            optimizer.zero_grad()
                            ave_loss.requires_grad = True
                            ave_loss.backward()
                            optimizer.step()
                            total_loss = torch.zeros([1, paramstopredict], dtype=torch.float32)


                for i in range(paramstopredict):
                    running_loss[i] += loss_list[i].item()
                    if target_norm:
                        output[:, i] = output[:, i] * stdev[i] + means[i]
                        y[:, i] = y[:, i] * stdev[i] + means[i]
                print(
                    "{} : epoch {} batch {} Output in train {}, Target in train {},".format(phase, epoch_num, batch_idx,
                                                                                            output.detach(), y.detach()), flush=True)
                f.write(
                    "{} : epoch {} batch {} Output in train {} Target in train {}\n".format(phase, epoch_num, batch_idx,
                                                                                            output.detach(), y.detach()))


            per_sample_epoch_loss = list(map(lambda x: x / float(data_sizes[phase]), running_loss))
            epoch_loss = list(map(lambda x: x / data_sizes[phase], running_loss))
            loss_hist[phase].append(epoch_loss)

            if verbose or epoch_num % 1 == 0:
                print('Epoch: {}, Phase: {}, Total Epoch loss: {}, Per Sample Epoch Loss: {}'.format(epoch_num, phase, epoch_loss, per_sample_epoch_loss), flush=True)
                print('-' * 30, flush=True)
                f.write('Epoch: {}, Phase: {}, Total Epoch loss: {}, Per Sample Epoch Loss: {}\n'.format(epoch_num, phase, epoch_loss, per_sample_epoch_loss))
                f.write('-----------------------------------------------------------------\n')
            torch.save(model, folderpath + '/epoch{}{}'.format(phase, epoch_num))
            if phase == 'validate' and sum(epoch_loss) < sum(best_loss):
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(model, folderpath + '/lowest_error_model')

        if inclr and 30 < epoch_num <= 70 and not changed_lr:
            changed_lr = 1
            clip += 0.2
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01

        if inclr and epoch_num > 70 and changed_lr:
            changed_lr = 0
            clip = orig_clip
            for param_group in optimizer.param_groups:
                param_group['lr'] = orig_lr

        if epoch_num % 15 == 0 and epoch_num != 0:
            loss, correct = test(f, model, dataloader, loss_criterion_list, paramstopredict, means, stdev, target_norm,
                                 check_val=10)
            loss_per_sample = list(map(lambda x: x / float(data_sizes['test']), loss))
            print("Test : Loss: {} Correct: {} Loss per Sample: {}\n".format(loss, correct, loss_per_sample))
            f.write("Test : Loss: {} Correct: {} Loss per Sample: {}\n".format(loss, correct, loss_per_sample))

        if sch:
            scheduler.step()
        for param in range(paramstopredict):
            logger.add_scalars('{}'.format(paramlist[param]),
                               {'Training loss': loss_hist['train'][epoch_num][param],
                                'Validation loss': loss_hist['validate'][epoch_num][param]
                                },
                               epoch_num)
            logger.add_scalars('Average Loss {}'.format(paramlist[param]),
                               {'Training Average loss': float(loss_hist['train'][epoch_num][param]) / float(
                                   data_sizes['train']),
                                'Validation Average loss': float(loss_hist['validate'][epoch_num][param]) / float(
                                    data_sizes['validate'])
                                },
                               epoch_num)

        logger.add_scalars('Loss for all',
                           {'Training Average loss': sum(loss_hist['train'][epoch_num]),
                            'Validation Average loss': sum(loss_hist['validate'][epoch_num])},
                           epoch_num)
        logger.add_scalars('Average Loss for all',
                           {'Training loss': float(sum(loss_hist['train'][epoch_num])) / float(data_sizes['train']),
                            'Validation loss': float(sum(loss_hist['validate'][epoch_num])) / float(
                                data_sizes['validate'])},
                           epoch_num)

    per_sample_best_loss = list(map(lambda x: x / float(data_sizes['validate']), best_loss))
    print('-' * 50, flush=True)
    print('Best validation loss: {}, Per Sample {}'.format(best_loss, per_sample_best_loss), flush=True)
    f.write('Best validation loss: {} , Per Sample {} \n'.format(best_loss, per_sample_best_loss))
    model.load_state_dict(best_model_wts)
    return model, loss_hist


def run(config):
    loss_dict = {1: nn.MSELoss(), 2: nn.SmoothL1Loss()}
    loss_criterion_list = []

    for i in range(config.paramstopredict):
        loss_criterion_list.append(loss_dict[1])

    pwd = os.getcwd()
    folder_path = pwd + '/' + config.expname
    os.makedirs(folder_path, exist_ok=True)

    means = []
    stdev = []

    log_path = folder_path + '/log'
    shutil.rmtree(log_path, ignore_errors=True)
    logger = SummaryWriter(log_dir=log_path)

    file = open(folder_path + '/results_test.txt', 'w+')
    file.write(folder_path+'\n')

    print(folder_path, flush=True)
    print(config, flush=True)
    file.write('\n')

    with open("subjects") as f:
        content = f.readlines()
    subject_inds = content[0].split(' ')
    subject_inds[-1] = subject_inds[-1].split('\n')[0]

    subject_train = subject_inds[0:668]
    subject_val = subject_inds[668:891]
    subject_test = subject_inds[891:]

    labels = pd.read_csv('merged_target.csv', skipinitialspace=True)
    s = [pd.Series(labels[config.targetparams[i]]) for i in range(len(config.targetparams))]

    s = [s[i].fillna(0) for i in range(len(config.targetparams))]
    if config.target_norm:
        means = [stat.mean(s[i]) for i in range(len(config.targetparams))]
        stdev = [stat.stdev(s[i]) for i in range(len(config.targetparams))]
        s = [(s[i] - means[i]) / stdev[i] for i in range(len(config.targetparams))]

    print(means, flush=True)
    print(stdev, flush=True)
    z = zip(*s)

    label_dict = dict(zip(labels['Subject'], z))
    transformed_dataset = {
        'train': BrainImages(config.datasetdir, label_dict, subject_train, prep=True, augment=True, train_data=True, rotation=config.irot, translation=True, gaussian=False, sub_mean_img = config.mean_image),
        'validate': BrainImages(config.datasetdir, label_dict, subject_val, prep=True, sub_mean_img = config.mean_image),
        'test': BrainImages(config.datasetdir, label_dict, subject_test, prep=True, sub_mean_img = config.mean_image)
    }

    dataloader = {x: DataLoader(transformed_dataset[x], batch_size=config.batchsize,
                                shuffle=True, num_workers=0) for x in ['train', 'validate', 'test']}
    data_sizes = {x: len(transformed_dataset[x]) for x in ['train', 'validate', 'test']}

    if config.modelnum == 0:
        model = models3d.resnet3d4(config.paramstopredict)
    elif config.modelnum == 1:
        model = models3d.resnet3d4bn(config.paramstopredict)
    elif config.modelnum == 2:
        model = models3d.net3d4bn(config.paramstopredict)
    elif config.modelnum == 3:
        model = models3d.resnet3d4smallfc(config.paramstopredict)
    elif config.modelnum == 4:
        model = models3d.net3d6smallfcbn(config.paramstopredict)
    elif config.modelnum == 5:
        model = models3d.net3d6smallfc(config.paramstopredict)
    elif config.modelnum == 6:
        model = models3d.resnet3d4smallfcbn(config.paramstopredict)
    elif config.modelnum == 7:
        model = models3d.net3d4smallfcbn(config.paramstopredict)
    elif config.modelnum == 8:
        model = models3d.net3d4(config.paramstopredict)
    elif config.modelnum == 9:
        model = models3d.net3d5smallfcbn(config.paramstopredict)
    elif config.modelnum == 10:
        model = models3d.net3d5smallfc(config.paramstopredict)
    elif config.modelnum == 11:
        model = models3d.resnet3d5(config.paramstopredict)
    elif config.modelnum == 12:
        model = models3d.resnet3d5smallfc(config.paramstopredict)
    elif config.modelnum == 13:
        model = models3d.resnet3d5smallfcbn(config.paramstopredict)
    elif config.modelnum == 14:
        model = models3d.resnet3d5bn(config.paramstopredict)
    elif config.modelnum == 15:
        model = models3d.net3d5bn(config.paramstopredict)
    elif config.modelnum == 16:
        model = models3d.net3dpapersmallfc(config.paramstopredict)
    elif config.modelnum == 17:
        model = models3d.resnet3d6smallfcbn(config.paramstopredict)
    elif config.modelnum == 18:
        model = models3d.resnet3d6smallfc(config.paramstopredict)
    elif config.modelnum == 19:
        model = models3d.resnet3dpapersmallfc(config.paramstopredict)
    elif config.modelnum == 20:
        model = models3d.net3d6bn(config.paramstopredict)
    elif config.modelnum == 21:
        model = models3d.net3dpaper(config.paramstopredict)
    elif config.modelnum == 22:
        model = models3d.resnet3dpaper(config.paramstopredict)
    elif config.modelnum == 23:
        model = models3d.resnet3d6bn(config.paramstopredict)
    elif config.modelnum == 24:
        model = models3d.resnet3d6(config.paramstopredict)
    elif config.modelnum == 25:
        model = models3d.net3d5notebook(config.paramstopredict)

    model.apply(weights_init)
    model.float().to(device)

    if config.optimizer == 0:
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weightdecay)
    elif config.optimizer == 1:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weightdecay, momentum=config.mval)
    else:
        optim.RMSprop(model.parameters(), lr=config.lr, alpha=0.99, eps=1e-08, weight_decay=config.weightdecay, momentum=config.mval, centered=False)

    scheduler = optim.lr_scheduler.StepLR(optimizer, config.scheduler_step, config.gamma)

    model, loss_hist = train(config.lr, config.targetparams,config.target_norm, logger, config.inclr, config.rbs, config.sch, scheduler, config.clip, file, model, optimizer, loss_criterion_list, dataloader, data_sizes, config.epochs, folder_path,
                             config.paramstopredict,means, stdev, verbose=False)

    loss, correct = test(file, model, dataloader, loss_criterion_list, config.paramstopredict, means, stdev, config.target_norm,
                         config.acc_check)

    loss_per_sample = list(map(lambda x: x / float(data_sizes['test']), loss))
    print("Loss {}".format(loss))
    print("Correct {}".format(correct))
    correct = list(map(lambda x: float(x), correct))
    ct = list(map(lambda x: x / float(data_sizes['test']), correct))
    file.write("---------------------------------------------------------\n")
    file.write("Total Test loss: {}, Total within {}: {}/{}, Total per sample loss: {}\n".format(loss, config.acc_check, correct, data_sizes['test'], loss_per_sample))
    print("Total loss: {} Total within {}: {}/{}, Total per sample loss: {}".format(loss, config.acc_check, correct, data_sizes['test'], loss_per_sample), flush=True)
    print("C/T: {}".format(ct), flush=True)
    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expname", help="Name of the experiment")
    parser.add_argument("--datasetdir", default='/home/cy1235/mlhc', type=str,
                        help="Directory name excluding top_to_bottom etc")
    parser.add_argument("--paramstopredict", default=1, type=int, help="Number of parameters to predict")
    parser.add_argument('-p', '--targetparams', nargs='+', type=str, help='<Required> Parameters to predict',
                        required=True)
    #parser.add_argument('--loss_list', nargs='+', type=int, help='<Required> Losses to use',required=True)
    parser.add_argument("--lr", default=0.0001, type=float, help="Input Learning Rate")
    parser.add_argument("--batchsize", default=1, type=int, help="Batch size for training")
    parser.add_argument("--acc_check", default=2, type=int, help="Accuracy checking for test")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("--init", action='store_true', help="Whether to initialize weights?")
    parser.add_argument("--target_norm", action='store_true', help="Whether to do normalization?")
    parser.add_argument("--scheduler_step", default=10, type=int, help="Step LR after how many steps?")
    parser.add_argument("--gamma", default=0.1, type=float, help="Scheduler gamma")
    parser.add_argument("--mval", default=0, type=float, help="momentum value") #0.9
    parser.add_argument("--irot", action='store_true', help="Whether to include rotation in augmentation")
    parser.add_argument("--clip", default=0.5, type=float, help="Whether or not to clip")
    parser.add_argument("--weightdecay", default=3e-6, type=float, help="Whether to decay weight") #0.0001
    parser.add_argument("--modelnum", default=0, type=int, help="Which model to choose")
    parser.add_argument("--optimizer", default=0, type=int, help="Which optimizer to choose")
    parser.add_argument("--sch", action='store_true', help="Whether to do scheduling of LR")
    parser.add_argument("--rbs", default=15, type=int, help="Real batchsize for accumulating")
    parser.add_argument("--inclr", action='store_true', help="Whether or not to increase LR")
    parser.add_argument("--mean_image", action='store_true', help="Whether to subtract mean image")
    config = parser.parse_args()
    run(config)
