import argparse
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from dataset import BrainImages
import models2d
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


def test(f, model, dataloader, loss_criterion_list, paramstopredict, means, stdev, target_prep, check_val=3):
    with torch.no_grad():
        model.eval()
        total_loss_list = [0] * paramstopredict
        total_correct_list = [0] * paramstopredict
        for batch_no, data in enumerate(dataloader['test']):
            x = data['x'].to(device)
            y = data['y'].to(device)
            predicted = model(x)
            for i in range(paramstopredict):
                total_loss_list[i] += loss_criterion_list[i](predicted[:, i], y[:, i])
                if target_prep:
                    predicted[:, i] = predicted[:, i] * stdev[i] + means[i]
                    y[:, i] = y[:, i] * stdev[i] + means[i]
            print("Test : Predicted Desired {} {}".format(predicted, y), flush=True)
            f.write("Test : Predicted Desired {} {}\n".format(predicted, y))
            for i in range(paramstopredict):
                a = np.abs(np.around(np.array(predicted[:, i].detach()), decimals=3))
                b = np.abs(np.around(np.array(y[:, i].detach()), decimals=3))
                total_correct_list[i] += (np.abs(a - b) <= check_val).sum()
    return total_loss_list, total_correct_list


def train(lr, logger, paramlist, sch, scheduler, inclr, f, clip, target_norm,  model, optimizer, loss_criterion_list, dataloader, data_sizes, means, stdev,num_epochs, folderpath, paramstopredict,
          verbose=False):
    best_loss = np.full(paramstopredict, np.inf).tolist()
    loss_hist = {'train': [], 'validate': []}
    orig_lr = lr
    orig_clip = clip
    changed_lr = 0
    for epoch_num in range(num_epochs):
        for phase in ['train', 'validate']:
            running_loss = [0] * paramstopredict
            if phase == 'train':
                model.train(True)
            else:
                model.eval()

            for batch_idx, data in enumerate(dataloader[phase]):
                loss_list = []
                optimizer.zero_grad()
                x = data['x'].to(device)
                y = data['y'].to(device)
                output = model(x)

                for i in range(paramstopredict):
                    loss_list.append(loss_criterion_list[i](output[:, i], y[:, i]))

                loss = sum(loss_list)
                torch.nn.utils.clip_grad_value_(model.parameters(), clip)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                for i in range(paramstopredict):
                    running_loss[i] += loss_list[i].item()
                    if target_norm:
                        output[:, i] = output[:, i] * stdev[i] + means[i]
                        y[:, i] = y[:, i] * stdev[i] + means[i]

                print("{} : epoch {} batch {} Output in train {}, Target in train {},".format(phase, epoch_num,
                                                                                              batch_idx, output.detach(),
                                                                                              y.detach()), flush=True)
                f.write("{} : epoch {} batch {} Output in train {} Target in train {}\n".format(phase, epoch_num,
                                                                                            batch_idx, output.detach(),
                                                                                            y.detach()))


            per_sample_epoch_loss = list(map(lambda x: x/float(data_sizes[phase]), running_loss))
            epoch_loss = list(map(lambda x: x, running_loss))
            loss_hist[phase].append(epoch_loss)
            if verbose or epoch_num % 1 == 0:
                print('Epoch: {}, Phase: {}, Total Epoch loss: {}, Per Sample Epoch Loss: {}'.format(epoch_num, phase, epoch_loss,per_sample_epoch_loss), flush=True)
                print('-' * 10, flush=True)
                f.write('Epoch: {}, Phase: {}, Total Epoch loss: {}, Per Sample Epoch Loss: {}\n'.format(epoch_num, phase,
                                                                                                     epoch_loss,
                                                                                                     per_sample_epoch_loss))
                f.write('-------------------------------------------\n')
            torch.save(model, folderpath + '/epoch{}{}'.format(phase, epoch_num))
            if phase == 'validate' and sum(epoch_loss) < sum(best_loss):
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(model, folderpath + '/lowest_error_model')
        if inclr and 30 < epoch_num <= 50 and not changed_lr:
            changed_lr = 1
            clip += 0.2
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01

        if inclr and epoch_num>50 and changed_lr:
            changed_lr = 0
            clip = orig_clip
            for param_group in optimizer.param_groups:
                param_group['lr'] = orig_lr

        if sch:
            scheduler.step()
        for param in range(paramstopredict):
            logger.add_scalars('{}'.format(paramlist[param]),
                               {'Training loss': loss_hist['train'][epoch_num][param],
                                'Validation loss': loss_hist['validate'][epoch_num][param]
                                },
                               epoch_num)
            logger.add_scalars('Average Loss {}'.format(paramlist[param]),
                               {'Training Average loss': float(loss_hist['train'][epoch_num][param])/float(data_sizes['train']),
                                'Validation Average loss': float(loss_hist['validate'][epoch_num][param])/float(data_sizes['validate'])
                                },
                               epoch_num)

        logger.add_scalars('Loss for all',
                           {'Training Average loss': sum(loss_hist['train'][epoch_num]),
                            'Validation Average loss': sum(loss_hist['validate'][epoch_num])},
                           epoch_num)
        logger.add_scalars('Average Loss for all',
                           {'Training loss': float(sum(loss_hist['train'][epoch_num]))/float(data_sizes['train']),
                            'Validation loss': float(sum(loss_hist['validate'][epoch_num]))/float(data_sizes['validate'])},
                           epoch_num)

    per_sample_best_loss = list(map(lambda x: x / float(data_sizes['validate']), best_loss))
    print('-' * 50, flush=True)
    f.write('Best validation loss: {} , Per Sample {} \n'.format(best_loss, per_sample_best_loss))
    print('Best validation loss: {}, Per Sample {}'.format(best_loss, per_sample_best_loss), flush=True)
    model.load_state_dict(best_model_wts)
    return model, loss_hist


def run(config):
    loss_dict = {1: nn.MSELoss(), 2:nn.SmoothL1Loss() }
    loss_criterion_list = []
    print(config.targetparams, flush=True)
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
    train_subjects = unpickling("train_subject_index")
    val_subjects = unpickling("val_subject_index")
    test_subjects = unpickling("test_subject_index")
    file_names = pd.read_csv("all_complete_path.csv")

    full_train_raw = list(file_names.iloc[train_subjects, 2])
    for i in range(len(full_train_raw)):
        full_train_raw[i] = full_train_raw[i].replace('/cbi/hcp/hcp_seg/aseg_all_slices', config.datasetdir)

    full_val_raw = list(file_names.iloc[val_subjects, 2])
    for i in range(len(full_val_raw)):
        full_val_raw[i] = full_val_raw[i].replace('/cbi/hcp/hcp_seg/aseg_all_slices', config.datasetdir)

    full_test_raw = list(file_names.iloc[test_subjects, 2])
    for i in range(len(full_test_raw)):
        full_test_raw[i] = full_test_raw[i].replace('/cbi/hcp/hcp_seg/aseg_all_slices', config.datasetdir)

    rand1 = np.arange(len(full_train_raw))
    np.random.shuffle(rand1)
    rand1 = rand1[:config.train_size]
    rand2 = np.arange(len(full_val_raw))
    np.random.shuffle(rand2)
    rand2 = rand2[:config.validate_size]
    rand3 = np.arange(len(full_test_raw))
    np.random.shuffle(rand3)
    rand3 = rand3[:config.test_size]

    labels = pd.read_csv('merged_target.csv', skipinitialspace=True)
    s = [pd.Series(labels[config.targetparams[i]]) for i in range(len(config.targetparams))]
    s = [s[i].fillna(0) for i in range(len(config.targetparams))]

    if config.target_norm:
        means = [stat.mean(s[i]) for i in range(len(config.targetparams))]
        stdev = [stat.stdev(s[i]) for i in range(len(config.targetparams))]
        s = [(s[i] - means[i]) / stdev[i] for i in range(len(config.targetparams))]
    print(folder_path, flush=True)
    print(config, flush=True)
    #file.write(config)
    file.write('\n')
    print(means, flush=True)
    print(stdev, flush=True)

    z = zip(*s)
    label_dict = dict(zip(labels['Subject'], z))

    subjects_train = [int(full_train_raw[i].split('/')[5]) for i in range(len(full_train_raw))]
    subjects_val = [int(full_val_raw[i].split('/')[5]) for i in range(len(full_val_raw))]
    subjects_test = [int(full_test_raw[i].split('/')[5]) for i in range(len(full_test_raw))]

    transformed_dataset = {
        'train': BrainImages(np.array(full_train_raw)[rand1], label_dict, subjects_train, prep=True, augment=True, train_data=True, rotation= config.irot, translation= True, mean_image = config.mean_image),
        'validate': BrainImages(np.array(full_val_raw)[rand2], label_dict, subjects_val, prep=True, mean_image = config.mean_image),
        'test': BrainImages(np.array(full_test_raw)[rand3], label_dict, subjects_test, prep=True, mean_image = config.mean_image)
    }

    dataloader = {x: DataLoader(transformed_dataset[x], batch_size=config.batchsize,
                                shuffle=True, num_workers=0) for x in ['train', 'validate', 'test']}
    data_sizes = {x: len(transformed_dataset[x]) for x in ['train', 'validate', 'test']}

    if config.modelnum==0:
        model = models2d.resnet2d4(config.paramstopredict)
    elif config.modelnum==1:
        model = models2d.resnet2d4fc256avg(config.paramstopredict)
    elif config.modelnum==2:
        model = models2d.resnet2d4fc256avgbn(config.paramstopredict)
    elif config.modelnum==3:
        model = models2d.resnet2d4bn(config.paramstopredict)
    elif config.modelnum==4:
        model = models2d.net2d4bn(config.paramstopredict)
    elif config.modelnum==5:
        model = models2d.resnet2d5(config.paramstopredict)
    elif config.modelnum==6:
        model = models2d.resnet2d5bn(config.paramstopredict)
    elif config.modelnum==7:
        model = models2d.resnet2d5fc128bn(config.paramstopredict)
    elif config.modelnum==8:
        model = models2d.resnet2d5fc128(config.paramstopredict)
    elif config.modelnum==9:
        model = models2d.net2d5bn(config.paramstopredict)
    elif config.modelnum==10:
        model = models2d.net2d5(config.paramstopredict)
    elif config.modelnum==11:
        model = models2d.net2d5fcsmallbn(config.paramstopredict)
    elif config.modelnum==12:
        model = models2d.net2d5fcsmall(config.paramstopredict)
    elif config.modelnum==13:
        model = models2d.resnet2dpapersmallfc(config.paramstopredict)
    elif config.modelnum==14:
        model = models2d.resnet2dpapersmallfc(config.paramstopredict)
    elif config.modelnum==15:
        model = models2d.net2dpapersmallfc(config.paramstopredict)
    elif config.modelnum==16:
        model = models2d.net2dpaper(config.paramstopredict)
    elif config.modelnum==17:
        model = models2d.resnet2dpaper(config.paramstopredict)
    elif config.modelnum==18:
        model = models2d.resnet2dpapersmallfc(config.paramstopredict)
    elif config.modelnum==19:
        model = models2d.net2d6smallfcbn(config.paramstopredict)
    elif config.modelnum==20:
        model = models2d.net2d6bn(config.paramstopredict)
    elif config.modelnum==21:
        model = models2d.net2d6bnsmallfcv2(config.paramstopredict)
    elif config.modelnum==22:
        model = models2d.resnet2d6bn(config.paramstopredict)
    elif config.modelnum==23:
        model = models2d.resnet2d6smallfcbn(config.paramstopredict)

    model.apply(weights_init)
    model.float().to(device)
    if config.optimizer == 0:
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weightdecay)
    elif config.optimizer == 1:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weightdecay, momentum=config.mval)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=config.lr, alpha=0.99, eps=1e-08, weight_decay=config.weightdecay, momentum=config.mval, centered=False)

    scheduler = optim.lr_scheduler.StepLR(optimizer, config.scheduler_step, config.gamma)

    model, loss_hist = train(config.lr, logger, config.targetparams,config.sch, scheduler, config.inclr, file, config.clip, config.target_norm, model, optimizer, loss_criterion_list, dataloader, data_sizes, means, stdev, config.epochs, folder_path,
                             config.paramstopredict, verbose=False)
    loss, correct = test(file, model, dataloader, loss_criterion_list, config.paramstopredict, means, stdev, config.target_norm,
                         config.acc_check)

    loss_per_sample = list(map(lambda x: x / float(data_sizes['test']), loss))
    print("Loss {}".format(loss), flush=True)
    print("Correct {}".format(correct), flush=True)
    correct = list(map(lambda x: float(x), correct))
    ct = list(map(lambda x: x / float(data_sizes['test']), correct))
    file.write("---------------------------------------------------------\n")
    file.write(
        "Total Test loss: {} Total within {}: {}/{} Avg Test Loss: {}\n".format(loss, config.acc_check, correct, data_sizes['test'], loss_per_sample))
    print("Total Test loss: {} \nTotal within {}: {}/{} Avg Test Loss: {}".format(loss, config.acc_check, correct, data_sizes['test'], loss_per_sample), flush=True)
    print("C/T: {}".format(ct), flush=True)
    file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expname", help="Name of the experiment")
    parser.add_argument("--datasetdir", default='/home/cy1235/mlhc', type=str, help="Directory name excluding top_to_bottom etc")
    parser.add_argument("--paramstopredict", default=1, type=int, help="Number of parameters to predict")
    parser.add_argument('-p', '--targetparams', nargs='+', type=str, help='<Required> Parameters to predict',
                        required=True)
    parser.add_argument("--lr", default=0.001, type=float, help="Input Learning Rate")
    parser.add_argument("--batchsize", default=30, type=int, help="Batch size for training")
    parser.add_argument("--acc_check", default=20, type=int, help="Accuracy checking for test")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("--init", action='store_true', help="Whether to initialize weights?")
    parser.add_argument("--target_norm", action='store_true', help="Whether to do normalization?")
    parser.add_argument("--scheduler_step", default=10, type=int, help="Step LR after how many steps?")
    parser.add_argument("--gamma", default=0.1, type=float, help="Scheduler gamma")
    parser.add_argument("--mval", default=0.0, type=float, help="momentum value")
    parser.add_argument("--irot", action='store_true', help="Whether to include rotation in augmentation")
    parser.add_argument("--clip", default=1, type=float, help="Whether or not to clip")
    parser.add_argument("--weightdecay", default=3e-6, type=float, help="Whether to decay weight")  # 0.0001
    parser.add_argument("--modelnum", default=0, type=int, help="Which model to choose")
    parser.add_argument("--optimizer", default=0, type=int, help="Which optimizer to choose")
    parser.add_argument("--sch", action='store_true', help="Whether to do scheduling of LR")
    parser.add_argument("--inclr", action='store_true', help="Whether or not to increase LR")
    parser.add_argument("--mean_image", action='store_true', help="Whether to subtract mean image")
    parser.add_argument("--train_size", default=5000, type=int, help="Training set size")
    parser.add_argument("--test_size", default=30000, type=int, help="Testing set size")
    parser.add_argument("--validate_size", default=30000, type=int, help="Validation set size")
    config = parser.parse_args()
    run(config)
