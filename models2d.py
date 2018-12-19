import torch.nn as nn
from collections import OrderedDict


# was without dropout n batchnorm n residual connections
class resnet2d4(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2d4, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1), stride=(2, 2)))
        ]))

        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool2d((2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(64*14*14, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-3, :-3]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2

        conv_avg = self.avg(conv_out4)

        shaped = conv_avg.view(-1, 64*14*14)
        output = self.fc(shaped)
        return output


class resnet2d4fc256avg(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2d4fc256avg, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1), stride=(2, 2)))
        ]))

        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool2d((4, 4), stride=4))
        ]))

        self.avg2 = nn.Sequential(OrderedDict([
            ('avg2', nn.AvgPool2d((2, 2), stride=2))
        ]))

        self.avg3 = nn.Sequential(OrderedDict([
            ('avg3', nn.AvgPool2d((2, 2), stride=1))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1
        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-3, :-3]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2
        conv_avg = self.avg(conv_out4)
        conv_avg2 = self.avg2(conv_avg)
        conv_avg3 = self.avg3(conv_avg2)
        shaped = conv_avg3.view(-1, 64 * 2 * 2)
        output = self.fc(shaped)
        return output

class resnet2d4fc256avgbn(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2d4fc256avgbn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('bn1', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('bn2', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('bn3', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1), stride=(2, 2)))
        ]))

        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool2d((4, 4), stride=4))
        ]))

        self.avg2 = nn.Sequential(OrderedDict([
            ('avg2', nn.AvgPool2d((2, 2), stride=2))
        ]))

        self.avg3 = nn.Sequential(OrderedDict([
            ('avg3', nn.AvgPool2d((2, 2), stride=1))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1
        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-3, :-3]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2
        conv_avg = self.avg(conv_out4)
        conv_avg2 = self.avg2(conv_avg)
        conv_avg3 = self.avg3(conv_avg2)
        shaped = conv_avg3.view(-1, 64 * 2 * 2)
        output = self.fc(shaped)
        return output

class resnet2d4bn(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2d4bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('bn1', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('bn2', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('bn3', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1), stride=(2, 2)))
        ]))

        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool2d((2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(64*14*14, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-3, :-3]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2

        conv_avg = self.avg(conv_out4)

        shaped = conv_avg.view(-1, 64*14*14)
        output = self.fc(shaped)
        return output


# was without dropout n batchnorm n residual connections
class net2d4bn(nn.Module):
    def __init__(self, paramstopredict):
        super(net2d4bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('bn1', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2)),

            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('bn2', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2)),

            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('bn3', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2)),

            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))
            ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(64*14*14, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet1(img)
        shaped = conv_out.view(-1, 64*14*14)
        output = self.fc(shaped)
        return output


# was without dropout n batchnorm n residual connections
class resnet2d5(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2d5, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d(( 2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 128, (1, 1), stride=( 4, 4)))
        ]))
        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool2d((2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(128 * 36, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1
        res2 = self.res2(conv_out2)
        res2 = res2[:, :, 0:12, 0:12]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out5 = self.convnet5(conv_out4)
        conv_out5 = conv_out5 + res2
        conv_avg = self.avg(conv_out5)
        shaped = conv_avg.view(-1, 128 * 36)
        output = self.fc(shaped)
        return output


class resnet2d5bn(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2d5bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('bn1', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('bn2', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d(( 2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('bn3', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('bn5', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 128, (1, 1), stride=( 4, 4)))
        ]))
        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool2d((2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(128 * 36, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1
        res2 = self.res2(conv_out2)
        res2 = res2[:, :, 0:12, 0:12]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out5 = self.convnet5(conv_out4)
        conv_out5 = conv_out5 + res2
        conv_avg = self.avg(conv_out5)
        shaped = conv_avg.view(-1, 128 * 36)
        output = self.fc(shaped)
        return output


class resnet2d5fc128bn(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2d5fc128bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('bn1', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('bn2', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('bn3', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('bn5', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 128, (1, 1), stride=(4, 4)))
        ]))
        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool2d((4, 4), stride=4))
        ]))
        self.avg2 = nn.Sequential(OrderedDict([
            ('avg2', nn.AvgPool2d((2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(128, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, 0:12, 0:12]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out5 = self.convnet5(conv_out4)
        conv_out5 = conv_out5 + res2

        conv_avg = self.avg(conv_out5)
        conv_avg2 = self.avg2(conv_avg)
        shaped = conv_avg2.view(-1, 128)
        output = self.fc(shaped)
        return output


class resnet2d5fc128(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2d5fc128, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))]))

        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 128, (1, 1), stride=(4, 4)))
        ]))
        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool2d((4, 4), stride=4))
        ]))
        self.avg2 = nn.Sequential(OrderedDict([
            ('avg2', nn.AvgPool2d((2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(128, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, 0:12, 0:12]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out5 = self.convnet5(conv_out4)
        conv_out5 = conv_out5 + res2

        conv_avg = self.avg(conv_out5)
        conv_avg2 = self.avg2(conv_avg)
        shaped = conv_avg2.view(-1, 128)
        output = self.fc(shaped)
        return output


class net2d5bn(nn.Module):
    def __init__(self, paramstopredict):
        super(net2d5bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('bn1', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2)),

            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('bn2', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2)),

            ('c3', nn.Conv2d(16, 32, ( 3, 3))),
            ('bn3', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2)),

            ('c4', nn.Conv2d(32, 64, ( 3, 3))),
            ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d(( 2, 2), stride=2)),

            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('bn5', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(128*12*12, 5530)),
            ('drop', nn.Dropout(inplace=True)),
            ('frlu1', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(5530, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 128*12*12)
        output = self.fc(shaped)
        return output

class net2d5(nn.Module):
    def __init__(self, paramstopredict):
        super(net2d5, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2)),

            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2)),

            ('c3', nn.Conv2d(16, 32, ( 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2)),

            ('c4', nn.Conv2d(32, 64, ( 3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d(( 2, 2), stride=2)),

            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('relu5', nn.LeakyReLU(0.1))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(128*12*12, 5530)),
            ('drop', nn.Dropout(inplace=True)),
            ('frlu1', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(5530, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 128*12*12)
        output = self.fc(shaped)
        return output


class net2d5fcsmallbn(nn.Module):
    def __init__(self, paramstopredict):
        super(net2d5fcsmallbn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('bn1', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2)),

            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('bn2', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2)),

            ('c3', nn.Conv2d(16, 32, ( 3, 3))),
            ('bn3', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2)),

            ('c4', nn.Conv2d(32, 64, ( 3, 3))),
            ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d(( 2, 2), stride=2)),

            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('bn5', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
            ('avg1', nn.AvgPool2d((2, 2), stride=2))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(128*6*6, 2304)),
            ('drop1', nn.Dropout(inplace=True)),
            ('frlu1', nn.LeakyReLU(0.1)),

            ('f7', nn.Linear(2304, 1152)),
            ('drop2', nn.Dropout(inplace=True)),
            ('frlu2', nn.LeakyReLU(0.1)),

            ('f8', nn.Linear(1152, 526)),
            ('drop3', nn.Dropout(inplace=True)),
            ('frlu3', nn.LeakyReLU(0.1)),

            ('f9', nn.Linear(526, 263)),
            ('drop4', nn.Dropout(inplace=True)),
            ('frlu4', nn.LeakyReLU(0.1)),

            ('f10', nn.Linear(263, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 128*36)
        output = self.fc(shaped)
        return output


class net2d5fcsmall(nn.Module):
    def __init__(self, paramstopredict):
        super(net2d5fcsmall, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2)),

            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2)),

            ('c3', nn.Conv2d(16, 32, ( 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2)),

            ('c4', nn.Conv2d(32, 64, ( 3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d(( 2, 2), stride=2)),

            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('avg1', nn.AvgPool2d((2, 2), stride=2))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(128 * 6 * 6, 2304)),
            ('drop1', nn.Dropout(inplace=True)),
            ('frlu1', nn.LeakyReLU(0.1)),

            ('f7', nn.Linear(2304, 1152)),
            ('drop2', nn.Dropout(inplace=True)),
            ('frlu2', nn.LeakyReLU(0.1)),

            ('f8', nn.Linear(1152, 526)),
            ('drop3', nn.Dropout(inplace=True)),
            ('frlu3', nn.LeakyReLU(0.1)),

            ('f9', nn.Linear(526, 263)),
            ('drop4', nn.Dropout(inplace=True)),
            ('frlu4', nn.LeakyReLU(0.1)),

            ('f10', nn.Linear(263, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 128*36)
        output = self.fc(shaped)
        return output



class net2d6bn(nn.Module):
    def __init__(self, paramstopredict):
        super(net2d6bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('bn1', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2)),

            ('c2', nn.Conv2d(8, 16, ( 3, 3))),
            ('bn2', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2)),

            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('bn3', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2)),

            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2)),

            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('bn5', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
            ('max5', nn.MaxPool2d((2, 2), stride=2)),

            ('c6', nn.Conv2d(128, 256, (3, 3))),
            ('bn6', nn.BatchNorm2d(256, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool2d((2, 2), stride=2))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256 * 4, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 256*4)
        output = self.fc(shaped)
        return output


class net2d6bnsmallfcv2(nn.Module):
    def __init__(self, paramstopredict):
        super(net2d6bnsmallfcv2, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('bn1', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2)),

            ('c2', nn.Conv2d(8, 16, ( 3, 3))),
            ('bn2', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2)),

            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('bn3', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2)),

            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2)),

            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('bn5', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
            ('max5', nn.MaxPool2d((2, 2), stride=2)),

            ('c6', nn.Conv2d(128, 256, (3, 3))),
            ('bn6', nn.BatchNorm2d(256, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool2d((2, 2), stride=2))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256 * 4, 512)),
            ('drop1', nn.Dropout(inplace=True)),
            ('relufc1', nn.LeakyReLU(0.1))]))
        self.fc2 = nn.Sequential(OrderedDict([
            ('f7', nn.Linear(512, 256)),
            ('drop2', nn.Dropout(inplace=True)),
            ('relufc2', nn.LeakyReLU(0.1))]))
        self.fc3 = nn.Sequential(OrderedDict([
            ('f7', nn.Linear(256, 128)),
            ('drop3', nn.Dropout(inplace=True)),
            ('relufc3', nn.LeakyReLU(0.1)),
            ('f8', nn.Linear(128, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 256*4)
        output = self.fc(shaped)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

class net2d6smallfcbn(nn.Module):
    def __init__(self, paramstopredict):
        super(net2d6smallfcbn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('bn1', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2)),

            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('bn2', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2)),

            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('bn3', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2)),

            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2)),

            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('bn5', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
            ('max5', nn.MaxPool2d((2, 2), stride=2)),

            ('c6', nn.Conv2d(128, 256, (3, 3))),
            ('bn6', nn.BatchNorm2d(256, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool2d((2, 2), stride=2)),
            ('avg7', nn.AvgPool2d((2, 2), stride=2))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 256)
        output = self.fc(shaped)
        return output


class net2dpaper(nn.Module):
    def __init__(self, paramstopredict):
        super(net2dpaper, self).__init__()
        self.paramstopredict = paramstopredict
        self.convblock1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('c2', nn.Conv2d(8, 8, (3, 3))),
            ('bn2', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock2 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(8, 16, (3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('c4', nn.Conv2d(16, 16, (3, 3))),
            ('bn4', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock3 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(16, 32, (3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('c6', nn.Conv2d(32, 32, (3, 3))),
            ('bn6', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock4 = nn.Sequential(OrderedDict([
            ('c7', nn.Conv2d(32, 64, (3, 3))),
            ('relu7', nn.LeakyReLU(0.1)),
            ('c8', nn.Conv2d(64, 64, ( 3, 3))),
            ('bn8', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool2d(( 2, 2), stride=2))]))
        self.convblock5 = nn.Sequential(OrderedDict([
            ('c9', nn.Conv2d(64, 128, (3, 3))),
            ('relu9', nn.LeakyReLU(0.1)),
            ('c10', nn.Conv2d(128, 128, (3, 3), padding=(1, 0))),
            ('bn10', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu10', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool2d((2, 2), stride=2, padding=(1, 0)))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(3072, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out1 = self.convblock1(img)
        conv_out2 = self.convblock2(conv_out1)
        conv_out3 = self.convblock3(conv_out2)
        conv_out4 = self.convblock4(conv_out3)
        conv_out5 = self.convblock5(conv_out4)
        shaped = conv_out5.view(-1, 3072)
        output = self.fc(shaped)
        return output


class resnet2dpaper(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2dpaper, self).__init__()
        self.paramstopredict = paramstopredict
        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))
        self.convblock1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('c2', nn.Conv2d(8, 8, (3, 3))),
            ('bn2', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock2 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(8, 16, (3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('c4', nn.Conv2d(16, 16, (3, 3))),
            ('bn4', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1), stride=(4, 4)))
        ]))
        self.convblock3 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(16, 32, (3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('c6', nn.Conv2d(32, 32, (3, 3))),
            ('bn6', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock4 = nn.Sequential(OrderedDict([
            ('c7', nn.Conv2d(32, 64, (3, 3))),
            ('relu7', nn.LeakyReLU(0.1)),
            ('c8', nn.Conv2d(64, 64, (3, 3))),
            ('bn8', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock5 = nn.Sequential(OrderedDict([
            ('c9', nn.Conv2d(64, 128, (3, 3))),
            ('relu9', nn.LeakyReLU(0.1)),
            ('c10', nn.Conv2d(128, 128, (3, 3), padding=(1, 0))),
            ('bn10', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu10', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool2d((2, 2), stride=2, padding=(1, 0)))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(3072, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-3, :-3]
        conv_out1 = self.convblock1(img)
        conv_out2 = self.convblock2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-4, :-4]
        conv_out3 = self.convblock3(conv_out2)
        conv_out4 = self.convblock4(conv_out3)
        conv_out4 = conv_out4 + res2
        conv_out5 = self.convblock5(conv_out4)
        shaped = conv_out5.view(-1, 3072)
        output = self.fc(shaped)
        return output


class resnet2dpapersmallfc(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2dpapersmallfc, self).__init__()
        self.paramstopredict = paramstopredict
        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))
        self.convblock1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('c2', nn.Conv2d(8, 8, (3, 3))),
            ('bn2', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock2 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(8, 16, (3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('c4', nn.Conv2d(16, 16, (3, 3))),
            ('bn4', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1), stride=(4, 4)))
        ]))
        self.convblock3 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(16, 32, (3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('c6', nn.Conv2d(32, 32, (3, 3))),
            ('bn6', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock4 = nn.Sequential(OrderedDict([
            ('c7', nn.Conv2d(32, 64, (3, 3))),
            ('relu7', nn.LeakyReLU(0.1)),
            ('c8', nn.Conv2d(64, 64, (3, 3))),
            ('bn8', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock5 = nn.Sequential(OrderedDict([
            ('c9', nn.Conv2d(64, 128, (3, 3))),
            ('relu9', nn.LeakyReLU(0.1)),
            ('c10', nn.Conv2d(128, 128, (3, 3), padding=(1, 0))),
            ('bn10', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu10', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool2d((2, 2), stride=2, padding=(1, 0))),
            ('avg', nn.AvgPool2d((2, 2), stride=2))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(768, 389)),
            ('drop1', nn.Dropout(inplace=True)),
            ('relufc1', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(389, 150)),
            ('drop2', nn.Dropout(inplace=True)),
            ('relufc2', nn.LeakyReLU(0.1)),
            ('f8', nn.Linear(150, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-3, :-3]
        conv_out1 = self.convblock1(img)
        conv_out2 = self.convblock2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-4, :-4]
        conv_out3 = self.convblock3(conv_out2)
        conv_out4 = self.convblock4(conv_out3)
        conv_out4 = conv_out4 + res2

        conv_out5 = self.convblock5(conv_out4)
        shaped = conv_out5.view(-1, 768)
        output = self.fc(shaped)
        return output


class resnet2d6bn(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2d6bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))

        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('bn1', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2))]))
        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('bn2', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1), stride=(4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('bn3', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2))]))
        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))]))

        self.res3 = nn.Sequential(OrderedDict([
            ('res3', nn.Conv1d(64, 256, (1, 1), stride=(4, 4)))
        ]))

        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('bn5', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
            ('max5', nn.MaxPool2d((2, 2), stride=2))]))
        self.convnet6 = nn.Sequential(OrderedDict([
            ('c6', nn.Conv2d(128, 256, (3, 3))),
            ('bn6', nn.BatchNorm2d(256, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool2d((2, 2), stride=2))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256 * 4, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-2, :-2]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2

        res3 = self.res3(conv_out4)
        res3 = res3[:, :, :-2, :-2]
        conv_out5 = self.convnet5(conv_out4)
        conv_out6 = self.convnet6(conv_out5)
        conv_out6 = conv_out6 + res3

        shaped = conv_out6.view(-1, 256 * 4)
        output = self.fc(shaped)
        return output


class resnet2dpapersmallfc(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2dpapersmallfc, self).__init__()
        self.paramstopredict = paramstopredict
        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))
        self.convblock1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('c2', nn.Conv2d(8, 8, (3, 3))),
            ('bn2', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock2 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(8, 16, (3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('c4', nn.Conv2d(16, 16, (3, 3))),
            ('bn4', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1), stride=(4, 4)))
        ]))
        self.convblock3 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(16, 32, (3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('c6', nn.Conv2d(32, 32, (3, 3))),
            ('bn6', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock4 = nn.Sequential(OrderedDict([
            ('c7', nn.Conv2d(32, 64, (3, 3))),
            ('relu7', nn.LeakyReLU(0.1)),
            ('c8', nn.Conv2d(64, 64, (3, 3))),
            ('bn8', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock5 = nn.Sequential(OrderedDict([
            ('c9', nn.Conv2d(64, 128, (3, 3))),
            ('relu9', nn.LeakyReLU(0.1)),
            ('c10', nn.Conv2d(128, 128, (3, 3), padding=(1, 0))),
            ('bn10', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu10', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool2d((2, 2), stride=2, padding=(1, 0)))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(3072, self.paramstopredict)),

        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-3, :-3]
        conv_out1 = self.convblock1(img)
        conv_out2 = self.convblock2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-4, :-4]
        conv_out3 = self.convblock3(conv_out2)
        conv_out4 = self.convblock4(conv_out3)
        conv_out4 = conv_out4 + res2
        conv_out5 = self.convblock5(conv_out4)
        shaped = conv_out5.view(-1, 3072)
        output = self.fc(shaped)
        return output


class resnet2dpapersmallfc(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2dpapersmallfc, self).__init__()
        self.paramstopredict = paramstopredict
        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))
        self.convblock1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('c2', nn.Conv2d(8, 8, (3, 3))),
            ('bn2', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock2 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(8, 16, (3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('c4', nn.Conv2d(16, 16, (3, 3))),
            ('bn4', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1), stride=(4, 4)))
        ]))
        self.convblock3 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(16, 32, (3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('c6', nn.Conv2d(32, 32, (3, 3))),
            ('bn6', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock4 = nn.Sequential(OrderedDict([
            ('c7', nn.Conv2d(32, 64, (3, 3))),
            ('relu7', nn.LeakyReLU(0.1)),
            ('c8', nn.Conv2d(64, 64, (3, 3))),
            ('bn8', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock5 = nn.Sequential(OrderedDict([
            ('c9', nn.Conv2d(64, 128, (3, 3))),
            ('relu9', nn.LeakyReLU(0.1)),
            ('c10', nn.Conv2d(128, 128, (3, 3), padding=(1, 0))),
            ('bn10', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu10', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool2d((2, 2), stride=2, padding=(1, 0))),
            ('avg', nn.AvgPool2d((2, 2), stride=2, padding=(1, 0)))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(1024, 512)),
            ('drop6', nn.Dropout(inplace=True)),
            ('relufc6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(512, 256)),
            ('drop7', nn.Dropout(inplace=True)),
            ('relufc7', nn.LeakyReLU(0.1)),
            ('f8', nn.Linear(256, 128)),
            ('drop8', nn.Dropout(inplace=True)),
            ('relufc8', nn.LeakyReLU(0.1)),
            ('f9', nn.Linear(128, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-3, :-3]
        conv_out1 = self.convblock1(img)
        conv_out2 = self.convblock2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-4, :-4]
        conv_out3 = self.convblock3(conv_out2)
        conv_out4 = self.convblock4(conv_out3)
        conv_out4 = conv_out4 + res2
        conv_out5 = self.convblock5(conv_out4)

        shaped = conv_out5.view(-1, 1024)

        output = self.fc(shaped)
        return output


class net2dpapersmallfc(nn.Module):
    def __init__(self, paramstopredict):
        super(net2dpapersmallfc, self).__init__()
        self.paramstopredict = paramstopredict

        self.convblock1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('c2', nn.Conv2d(8, 8, (3, 3))),
            ('bn2', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock2 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(8, 16, (3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('c4', nn.Conv2d(16, 16, (3, 3))),
            ('bn4', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))]))

        self.convblock3 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(16, 32, (3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('c6', nn.Conv2d(32, 32, (3, 3))),
            ('bn6', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock4 = nn.Sequential(OrderedDict([
            ('c7', nn.Conv2d(32, 64, (3, 3))),
            ('relu7', nn.LeakyReLU(0.1)),
            ('c8', nn.Conv2d(64, 64, (3, 3))),
            ('bn8', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool2d((2, 2), stride=2))]))
        self.convblock5 = nn.Sequential(OrderedDict([
            ('c9', nn.Conv2d(64, 128, (3, 3))),
            ('relu9', nn.LeakyReLU(0.1)),
            ('c10', nn.Conv2d(128, 128, (3, 3), padding=(1, 0))),
            ('bn10', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu10', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool2d((2, 2), stride=2, padding=(1, 0))),
            ('avg', nn.AvgPool2d((2, 2), stride=2, padding=(1, 0)))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(1024, 512)),
            ('drop6', nn.Dropout(inplace=True)),
            ('relufc6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(512, 256)),
            ('drop7', nn.Dropout(inplace=True)),
            ('relufc7', nn.LeakyReLU(0.1)),
            ('f8', nn.Linear(256, 128)),
            ('drop8', nn.Dropout(inplace=True)),
            ('relufc8', nn.LeakyReLU(0.1)),
            ('f9', nn.Linear(128, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out1 = self.convblock1(img)
        conv_out2 = self.convblock2(conv_out1)

        conv_out3 = self.convblock3(conv_out2)
        conv_out4 = self.convblock4(conv_out3)
        conv_out5 = self.convblock5(conv_out4)

        shaped = conv_out5.view(-1, 1024)

        output = self.fc(shaped)
        return output


class resnet2d6smallfcbn(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet2d6smallfcbn, self).__init__()

        self.paramstopredict = paramstopredict
        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1), stride=(4, 4)))
        ]))
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 8, (3, 3))),
            ('bn1', nn.BatchNorm2d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool2d((2, 2), stride=2))]))
        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(8, 16, (3, 3))),
            ('bn2', nn.BatchNorm2d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool2d((2, 2), stride=2))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1), stride=(4, 4)))
        ]))
        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 32, (3, 3))),
            ('bn3', nn.BatchNorm2d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool2d((2, 2), stride=2))]))
        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(32, 64, (3, 3))),
            ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool2d((2, 2), stride=2))]))
        self.res3 = nn.Sequential(OrderedDict([
            ('res3', nn.Conv1d(64, 256, (1, 1), stride=(4, 4)))
        ]))
        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(64, 128, (3, 3))),
            ('bn5', nn.BatchNorm2d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
            ('max5', nn.MaxPool2d((2, 2), stride=2))]))
        self.convnet6 = nn.Sequential(OrderedDict([
            ('c6', nn.Conv2d(128, 256, (3, 3))),
            ('bn6', nn.BatchNorm2d(256, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool2d((2, 2), stride=2)),
            ('avg7', nn.AvgPool2d((2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(1024, 512)),
            ('drop6', nn.Dropout(inplace=True)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(512, 256)),
            ('drop7', nn.Dropout(inplace=True)),
            ('relu7', nn.LeakyReLU(0.1)),
            ('f8', nn.Linear(256, 128)),
            ('drop8', nn.Dropout(inplace=True)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('f9', nn.Linear(128, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1
        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-2, :-2]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2
        res3 = self.res3(conv_out4)
        res3 = res3[:, :, :-2, :-2]
        conv_out5 = self.convnet5(conv_out4)
        conv_out6 = self.convnet6(conv_out5)
        conv_out6 = conv_out6 + res3
        shaped = conv_out6.view(-1, 256 * 4)
        output = self.fc(shaped)
        return output