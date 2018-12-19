import torch.nn as nn
from collections import OrderedDict
# was without dropout n batchnorm n residual connections
class resnet3d4(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3d4, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1, 1), stride=(2, 2, 2)))
        ]))

        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool3d((2, 2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(62720, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-3, :-3, :-3]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2

        conv_avg = self.avg(conv_out4)

        shaped = conv_avg.view(-1, 62720)
        output = self.fc(shaped)
        return output


class resnet3d4bn(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3d4bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('bn3', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1, 1), stride=(2, 2, 2)))
        ]))

        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool3d((2, 2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(62720, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-3, :-3, :-3]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2

        conv_avg = self.avg(conv_out4)

        shaped = conv_avg.view(-1, 62720)
        output = self.fc(shaped)
        return output


# was without dropout n batchnorm n residual connections
class net3d4bn(nn.Module):
    def __init__(self, paramstopredict):
        super(net3d4bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('bn3', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))
            ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(62720, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 62720)
        output = self.fc(shaped)
        return output

class resnet3d4smallfc(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3d4smallfc, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1, 1), stride=(2, 2, 2)))
        ]))

        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool3d((4, 4, 4), stride=4)),
            ('avg2', nn.AvgPool3d((2, 2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(64*1*3*3, 288)),
            ('drop6', nn.Dropout(inplace=True)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(288, 144)),
            ('drop7', nn.Dropout(inplace=True)),
            ('relu7', nn.LeakyReLU(0.1)),
            ('f8', nn.Linear(144, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-3, :-3, :-3]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2

        conv_avg = self.avg(conv_out4)
        shaped = conv_avg.view(-1, 64*9)
        output = self.fc(shaped)
        return output

class net3d6smallfcbn(nn.Module):
    def __init__(self, paramstopredict):
        super(net3d6smallfcbn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('bn3', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('bn5', nn.BatchNorm3d(128, track_running_stats=False)),
            ('max5', nn.MaxPool3d((2, 2, 2), stride=2)),
            ('relu5', nn.LeakyReLU(0.1))]))
        self.convnet6 = nn.Sequential(OrderedDict([
            ('c6', nn.Conv3d(128, 256, (3, 3, 3), padding=(2,0,0))),
            ('bn6', nn.BatchNorm3d(256, track_running_stats=False)),
            ('max6', nn.MaxPool3d((2, 2, 2), stride=2)),
            ('relu6', nn.LeakyReLU(0.1))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256*1*2*2, 512)),
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
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out5 = self.convnet5(conv_out4)
        conv_out6 = self.convnet6(conv_out5)
        shaped = conv_out6.view(-1, 1024)
        output = self.fc(shaped)
        return output

class net3d6smallfc(nn.Module):
    def __init__(self, paramstopredict):
        super(net3d6smallfc, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('max5', nn.MaxPool3d((2, 2, 2), stride=2)),
            ('relu5', nn.LeakyReLU(0.1))]))
        self.convnet6 = nn.Sequential(OrderedDict([
            ('c6', nn.Conv3d(128, 256, (3, 3, 3), padding=(2,0,0))),
            ('max6', nn.MaxPool3d((2, 2, 2), stride=2)),
            ('relu6', nn.LeakyReLU(0.1))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256*1*2*2, 512)),
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
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out5 = self.convnet5(conv_out4)
        conv_out6 = self.convnet6(conv_out5)
        shaped = conv_out6.view(-1, 1024)
        output = self.fc(shaped)
        return output



class resnet3d4smallfcbn(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3d4smallfcbn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('bn3', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1, 1), stride=(2, 2, 2)))
        ]))

        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool3d((4, 4, 4), stride=4)),
            ('avg2', nn.AvgPool3d((2, 2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(64*1*3*3, 288)),
            ('drop6', nn.Dropout(inplace=True)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(288, 144)),
            ('drop7', nn.Dropout(inplace=True)),
            ('relu7', nn.LeakyReLU(0.1)),
            ('f8', nn.Linear(144, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-3, :-3, :-3]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2

        conv_avg = self.avg(conv_out4)
        shaped = conv_avg.view(-1, 64*9)
        output = self.fc(shaped)
        return output

class net3d4smallfcbn(nn.Module):
    def __init__(self, paramstopredict):
        super(net3d4smallfcbn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('bn3', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2)),
            ('avg', nn.AvgPool3d((4,4,4), stride=4))
            ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(64*1*3*3, 288)),
            ('drop6', nn.Dropout(inplace=True)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(288, 144)),
            ('drop7', nn.Dropout(inplace=True)),
            ('relu7', nn.LeakyReLU(0.1)),
            ('f8', nn.Linear(144, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet1(img)
        shaped = conv_out.view(-1, 64*1*3*3)
        output = self.fc(shaped)
        return output

class net3d4(nn.Module):
    def __init__(self, paramstopredict):
        super(net3d4, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2)),
            ('avg', nn.AvgPool3d((4,4,4), stride=4))
            ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(64*1*3*3, 288)),
            ('drop6', nn.Dropout(inplace=True)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(288, 144)),
            ('drop7', nn.Dropout(inplace=True)),
            ('relu7', nn.LeakyReLU(0.1)),
            ('f8', nn.Linear(144, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet1(img)
        shaped = conv_out.view(-1, 64*1*3*3)
        output = self.fc(shaped)
        return output

class net3d5smallfcbn(nn.Module):
    def __init__(self, paramstopredict):
        super(net3d5smallfcbn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('bn3', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('bn5', nn.BatchNorm3d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2)),
            ('avg1', nn.AvgPool3d((2,2,2), stride=2, padding=(1,0,0))),
            ('avg2', nn.AvgPool3d((2,2,2), stride=2, padding=(1,0,0)))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(2304, 1152)),
            ('drop', nn.Dropout3d(inplace=True)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(1152, 576)),
            ('drop2', nn.Dropout3d(inplace=True)),
            ('relu7', nn.LeakyReLU(0.1)),
            ('f8', nn.Linear(576, 288)),
            ('drop3', nn.Dropout3d(inplace=True)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('f9', nn.Linear(288, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 128*2*9)
        output = self.fc(shaped)
        return output

class net3d5smallfc(nn.Module):
    def __init__(self, paramstopredict):
        super(net3d5smallfc, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2)),
            ('avg1', nn.AvgPool3d((2,2,2), stride=2, padding=(1,0,0))),
            ('avg2', nn.AvgPool3d((2,2,2), stride=2, padding=(1,0,0)))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(2304, 1152)),
            ('drop6', nn.Dropout3d(inplace=True)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(1152, 576)),
            ('drop7', nn.Dropout3d(inplace=True)),
            ('relu7', nn.LeakyReLU(0.1)),
            ('f8', nn.Linear(576, 288)),
            ('drop8', nn.Dropout3d(inplace=True)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('f9', nn.Linear(288, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 128*2*9)
        output = self.fc(shaped)
        return output

# was without dropout n batchnorm n residual connections
class resnet3d5(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3d5, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 128, (1, 1, 1), stride=(4, 4, 4)))
        ]))
        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool3d((2, 2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(128 * 36, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1
        res2 = self.res2(conv_out2)
        res2 = res2[:, :, 0:3, 0:12, 0:12]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out5 = self.convnet5(conv_out4)
        conv_out5 = conv_out5 + res2
        conv_avg = self.avg(conv_out5)
        shaped = conv_avg.view(-1, 128 * 36)
        output = self.fc(shaped)
        return output

class resnet3d5smallfc(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3d5smallfc, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 128, (1, 1, 1), stride=(4, 4, 4)))
        ]))
        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool3d((2, 2, 2), stride=2, padding=(1,0,0))),
            ('avg2', nn.AvgPool3d((2, 2, 2), stride=2, padding=(1,0,0))),
            ('avg3', nn.AvgPool3d((2, 2, 2), stride=2, padding=(1,0,0)))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256, 128)),
            ('drop6', nn.Dropout(inplace=True)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(128, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, 0:3, 0:12, 0:12]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out5 = self.convnet5(conv_out4)
        conv_out5 = conv_out5 + res2

        conv_avg = self.avg(conv_out5)
        shaped = conv_avg.view(-1, 256)
        output = self.fc(shaped)
        return output


class resnet3d5smallfcbn(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3d5smallfcbn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('bn3', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('bn5', nn.BatchNorm3d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 128, (1, 1, 1), stride=(4, 4, 4)))
        ]))
        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool3d((2, 2, 2), stride=2, padding=(1, 0, 0))),
            ('avg2', nn.AvgPool3d((2, 2, 2), stride=2, padding=(1, 0, 0))),
            ('avg3', nn.AvgPool3d((2, 2, 2), stride=2, padding=(1, 0, 0)))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256, 128)),
            ('drop6', nn.Dropout(inplace=True)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(128, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, 0:3, 0:12, 0:12]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out5 = self.convnet5(conv_out4)
        conv_out5 = conv_out5 + res2

        conv_avg = self.avg(conv_out5)
        shaped = conv_avg.view(-1, 128 * 2)
        output = self.fc(shaped)
        return output

class resnet3d5bn(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3d5bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('bn3', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))

        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('bn5', nn.BatchNorm3d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 128, (1, 1, 1), stride=(4, 4, 4)))
        ]))
        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool3d((2, 2, 2), stride=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(128 * 36, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, 0:3, 0:12, 0:12]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out5 = self.convnet5(conv_out4)
        conv_out5 = conv_out5 + res2

        conv_avg = self.avg(conv_out5)
        shaped = conv_avg.view(-1, 128 * 36)
        output = self.fc(shaped)
        return output

class net3d5bn(nn.Module):
    def __init__(self, paramstopredict):
        super(net3d5bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('bn3', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('bn5', nn.BatchNorm3d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(128*3*12*12, 5530)),
            ('drop', nn.Dropout3d(inplace=True)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(5530, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 128*3*12*12)
        output = self.fc(shaped)
        return output

class net3dpapersmallfc(nn.Module):
    def __init__(self, paramstopredict):
        super(net3dpapersmallfc, self).__init__()
        self.paramstopredict = paramstopredict
        self.convblock1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('c2', nn.Conv3d(8, 8, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock2 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('c4', nn.Conv3d(16, 16, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock3 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('c6', nn.Conv3d(32, 32, (3, 3, 3))),
            ('bn6', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock4 = nn.Sequential(OrderedDict([
            ('c7', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu7', nn.LeakyReLU(0.1)),
            ('c8', nn.Conv3d(64, 64, (3, 3, 3))),
            ('bn8', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock5 = nn.Sequential(OrderedDict([
            ('c9', nn.Conv3d(64, 128, (3, 3, 3))),
            ('relu9', nn.LeakyReLU(0.1)),
            ('c10', nn.Conv3d(128, 128, (3, 3, 3), padding=(1, 0, 0))),
            ('bn10', nn.BatchNorm3d(128, track_running_stats=False)),
            ('relu10', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool3d((2, 2, 2), stride=2, padding=(1, 0, 0)))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(2048, 1024)),
            ('drop6', nn.Dropout3d(inplace=True)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('f7', nn.Linear(1024, 512)),
            ('drop7', nn.Dropout3d(inplace=True)),
            ('relu7', nn.LeakyReLU(0.1)),
            ('f8', nn.Linear(512, 256)),
            ('drop8', nn.Dropout3d(inplace=True)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('f9', nn.Linear(256, paramstopredict))
        ]))

    def forward(self, img):
        conv_out1 = self.convblock1(img)
        conv_out2 = self.convblock2(conv_out1)
        conv_out3 = self.convblock3(conv_out2)
        conv_out4 = self.convblock4(conv_out3)
        conv_out5 = self.convblock5(conv_out4)
        shaped = conv_out5.view(-1, 2048)
        output = self.fc(shaped)
        return output


class resnet3d6smallfcbn(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3d6smallfcbn, self).__init__()

        self.paramstopredict = paramstopredict
        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1, 1), stride=(4, 4, 4)))
        ]))
        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('bn3', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.res3 = nn.Sequential(OrderedDict([
            ('res3', nn.Conv1d(64, 256, (1, 1, 1), stride=6))
        ]))
        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('bn5', nn.BatchNorm3d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),
            ('max5', nn.MaxPool3d((2, 2, 2), stride=2, padding=(1, 0, 0)))
        ]))

        self.convnet6 = nn.Sequential(OrderedDict([
            ('c6', nn.Conv3d(128, 256, (3, 3, 3), padding=(1, 0, 0))),
            ('bn6', nn.BatchNorm3d(256, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.AvgPool3d((2, 2, 2), stride=2))
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
        res1 = res1[:, :, :-2, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-2, :-2, :-2]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2

        res3 = self.res3(conv_out4)
        res3 = res3[:, :, :, :-1, :-1]
        conv_out5 = self.convnet5(conv_out4)
        conv_out6 = self.convnet6(conv_out5)
        conv_out6 = conv_out6 + res3
        shaped = conv_out6.view(-1, 1024)
        output = self.fc(shaped)
        return output


class resnet3d6smallfc(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3d6smallfc, self).__init__()

        self.paramstopredict = paramstopredict
        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1, 1), stride=(4, 4, 4)))
        ]))
        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.res3 = nn.Sequential(OrderedDict([
            ('res3', nn.Conv1d(64, 256, (1, 1, 1), stride=6))
        ]))
        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('max5', nn.MaxPool3d((2, 2, 2), stride=2, padding=(1, 0, 0)))
        ]))

        self.convnet6 = nn.Sequential(OrderedDict([
            ('c6', nn.Conv3d(128, 256, (3, 3, 3), padding=(1, 0, 0))),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.AvgPool3d((2, 2, 2), stride=2))
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
        res1 = res1[:, :, :-2, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-2, :-2, :-2]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2

        res3 = self.res3(conv_out4)
        res3 = res3[:, :, :, :-1, :-1]
        conv_out5 = self.convnet5(conv_out4)
        conv_out6 = self.convnet6(conv_out5)
        conv_out6 = conv_out6 + res3
        shaped = conv_out6.view(-1, 1024)
        output = self.fc(shaped)
        return output

class resnet3dpapersmallfc(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3dpapersmallfc, self).__init__()
        self.paramstopredict = paramstopredict
        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))

        self.convblock1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('c2', nn.Conv3d(8, 8, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock2 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('c4', nn.Conv3d(16, 16, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 128, (1, 1, 1), stride=(8, 8, 8)))
        ]))
        self.convblock3 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('c6', nn.Conv3d(32, 32, (3, 3, 3))),
            ('bn6', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock4 = nn.Sequential(OrderedDict([
            ('c7', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu7', nn.LeakyReLU(0.1)),
            ('c8', nn.Conv3d(64, 64, (3, 3, 3))),
            ('bn8', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock5 = nn.Sequential(OrderedDict([
            ('c9', nn.Conv3d(64, 128, (3, 3, 3))),
            ('relu9', nn.LeakyReLU(0.1)),
            ('c10', nn.Conv3d(128, 128, (3, 3, 3), padding=(1, 0, 0))),
            ('bn10', nn.BatchNorm3d(128, track_running_stats=False)),
            ('relu10', nn.LeakyReLU(0.1))
        ]))

        self.avg = nn.Sequential(OrderedDict([
            ('avg1', nn.AvgPool3d((2, 2, 2), stride=2, padding=(1, 0, 0))),
            ('avg2', nn.AvgPool3d((2, 2, 2), stride=2, padding=(1, 0, 0)))
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
        res1 = res1[:, :, :-4, :-3, :-3]
        conv_out1 = self.convblock1(img)
        conv_out2 = self.convblock2(conv_out1)
        conv_out2 = conv_out2 + res1
        res2 = self.res2(conv_out2)
        conv_out3 = self.convblock3(conv_out2)
        conv_out4 = self.convblock4(conv_out3)
        conv_out5 = self.convblock5(conv_out4)
        conv_out5 = conv_out5 + res2
        avg = self.avg(conv_out5)
        shaped = avg.view(-1, 1024)
        output = self.fc(shaped)
        return output


class net3d6bn(nn.Module):
    def __init__(self, paramstopredict):
        super(net3d6bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('bn3', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('bn5', nn.BatchNorm3d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1)),

            ('c6', nn.Conv3d(128, 256, (3, 3, 3))),
            ('bn6', nn.BatchNorm3d(256, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256 * 1 * 10 * 10, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 25600)
        output = self.fc(shaped)
        return output


class net3dpaper(nn.Module):
    def __init__(self, paramstopredict):
        super(net3dpaper, self).__init__()
        self.paramstopredict = paramstopredict
        self.convblock1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('c2', nn.Conv3d(8, 8, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock2 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('c4', nn.Conv3d(16, 16, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock3 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('c6', nn.Conv3d(32, 32, (3, 3, 3))),
            ('bn6', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock4 = nn.Sequential(OrderedDict([
            ('c7', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu7', nn.LeakyReLU(0.1)),
            ('c8', nn.Conv3d(64, 64, (3, 3, 3))),
            ('bn8', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock5 = nn.Sequential(OrderedDict([
            ('c9', nn.Conv3d(64, 128, (3, 3, 3))),
            ('relu9', nn.LeakyReLU(0.1)),
            ('c10', nn.Conv3d(128, 128, (3, 3, 3), padding=(1, 0, 0))),
            ('bn10', nn.BatchNorm3d(128, track_running_stats=False)),
            ('relu10', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool3d((2, 2, 2), stride=2, padding=(1, 0, 0)))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(2048, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out1 = self.convblock1(img)
        conv_out2 = self.convblock2(conv_out1)
        conv_out3 = self.convblock3(conv_out2)
        conv_out4 = self.convblock4(conv_out3)
        conv_out5 = self.convblock5(conv_out4)
        shaped = conv_out5.view(-1, 2048)
        output = self.fc(shaped)
        return output


class resnet3dpaper(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3dpaper, self).__init__()
        self.paramstopredict = paramstopredict
        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))

        self.convblock1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('c2', nn.Conv3d(8, 8, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock2 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('c4', nn.Conv3d(16, 16, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 128, (1, 1, 1), stride=(8, 8, 8)))
        ]))
        self.convblock3 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('c6', nn.Conv3d(32, 32, (3, 3, 3))),
            ('bn6', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1)),
            ('max6', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock4 = nn.Sequential(OrderedDict([
            ('c7', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu7', nn.LeakyReLU(0.1)),
            ('c8', nn.Conv3d(64, 64, (3, 3, 3))),
            ('bn8', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu8', nn.LeakyReLU(0.1)),
            ('max8', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convblock5 = nn.Sequential(OrderedDict([
            ('c9', nn.Conv3d(64, 128, (3, 3, 3))),
            ('relu9', nn.LeakyReLU(0.1)),
            ('c10', nn.Conv3d(128, 128, (3, 3, 3), padding=(1, 0, 0))),
            ('bn10', nn.BatchNorm3d(128, track_running_stats=False)),
            ('relu10', nn.LeakyReLU(0.1))
        ]))

        self.avg = nn.Sequential(OrderedDict([
            ('avg', nn.AvgPool3d((2, 2, 2), stride=2, padding=(1, 0, 0)))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(4096, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-4, :-3, :-3]
        conv_out1 = self.convblock1(img)
        conv_out2 = self.convblock2(conv_out1)
        conv_out2 = conv_out2 + res1
        res2 = self.res2(conv_out2)
        conv_out3 = self.convblock3(conv_out2)
        conv_out4 = self.convblock4(conv_out3)
        conv_out5 = self.convblock5(conv_out4)
        conv_out5 = conv_out5 + res2
        avg = self.avg(conv_out5)
        shaped = avg.view(-1, 4096)
        output = self.fc(shaped)
        return output


class resnet3d6bn(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3d6bn, self).__init__()

        self.paramstopredict = paramstopredict
        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('bn1', nn.BatchNorm3d(8, track_running_stats=False)),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('bn2', nn.BatchNorm3d(16, track_running_stats=False)),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1, 1), stride=(4, 4, 4)))
        ]))
        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('bn3', nn.BatchNorm3d(32, track_running_stats=False)),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('bn4', nn.BatchNorm3d(64, track_running_stats=False)),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.res3 = nn.Sequential(OrderedDict([
            ('res3', nn.Conv1d(64, 256, (1, 1, 1), stride=(1, 1, 1)))
        ]))
        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('bn5', nn.BatchNorm3d(128, track_running_stats=False)),
            ('relu5', nn.LeakyReLU(0.1))]))

        self.convnet6 = nn.Sequential(OrderedDict([
            ('c6', nn.Conv3d(128, 256, (3, 3, 3))),
            ('bn6', nn.BatchNorm3d(256, track_running_stats=False)),
            ('relu6', nn.LeakyReLU(0.1))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256 * 1 * 10 * 10, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-2, :-2, :-2]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2

        res3 = self.res3(conv_out4)
        res3 = res3[:, :, :-4, :-4, :-4]
        conv_out5 = self.convnet5(conv_out4)
        conv_out6 = self.convnet6(conv_out5)
        conv_out6 = conv_out6 + res3
        shaped = conv_out6.view(-1, 25600)
        output = self.fc(shaped)
        return output


class resnet3d6(nn.Module):
    def __init__(self, paramstopredict):
        super(resnet3d6, self).__init__()
        self.paramstopredict = paramstopredict
        self.res1 = nn.Sequential(OrderedDict([
            ('res1', nn.Conv1d(1, 16, (1, 1, 1), stride=(4, 4, 4)))
        ]))
        self.convnet1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('res2', nn.Conv1d(16, 64, (1, 1, 1), stride=(4, 4, 4)))
        ]))
        self.convnet3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.convnet4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2))]))
        self.res3 = nn.Sequential(OrderedDict([
            ('res3', nn.Conv1d(64, 256, (1, 1, 1), stride=(1, 1, 1)))
        ]))
        self.convnet5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('relu5', nn.LeakyReLU(0.1))]))

        self.convnet6 = nn.Sequential(OrderedDict([
            ('c6', nn.Conv3d(128, 256, (3, 3, 3))),
            ('relu6', nn.LeakyReLU(0.1))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256 * 1 * 10 * 10, self.paramstopredict))
        ]))

    def forward(self, img):
        res1 = self.res1(img)
        res1 = res1[:, :, :-2, :-2, :-2]
        conv_out1 = self.convnet1(img)
        conv_out2 = self.convnet2(conv_out1)
        conv_out2 = conv_out2 + res1

        res2 = self.res2(conv_out2)
        res2 = res2[:, :, :-2, :-2, :-2]
        conv_out3 = self.convnet3(conv_out2)
        conv_out4 = self.convnet4(conv_out3)
        conv_out4 = conv_out4 + res2

        res3 = self.res3(conv_out4)
        res3 = res3[:, :, :-4, :-4, :-4]
        conv_out5 = self.convnet5(conv_out4)
        conv_out6 = self.convnet6(conv_out5)
        conv_out6 = conv_out6 + res3
        shaped = conv_out6.view(-1, 25600)
        output = self.fc(shaped)
        return output


class net3d5notebook(nn.Module):
    def __init__(self, paramstopredict):
        super(net3d5notebook, self).__init__()

        self.paramstopredict = paramstopredict
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv3d(1, 8, (3, 3, 3))),
            ('relu1', nn.LeakyReLU(0.1)),
            ('max1', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c2', nn.Conv3d(8, 16, (3, 3, 3))),
            ('relu2', nn.LeakyReLU(0.1)),
            ('max2', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c3', nn.Conv3d(16, 32, (3, 3, 3))),
            ('relu3', nn.LeakyReLU(0.1)),
            ('max3', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c4', nn.Conv3d(32, 64, (3, 3, 3))),
            ('relu4', nn.LeakyReLU(0.1)),
            ('max4', nn.MaxPool3d((2, 2, 2), stride=2)),

            ('c5', nn.Conv3d(64, 128, (3, 3, 3))),
            ('relu5', nn.LeakyReLU(0.1)),
            ('max5', nn.MaxPool3d((2, 2, 2), stride=2))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(4608, self.paramstopredict))
        ]))

    def forward(self, img):
        conv_out = self.convnet(img)
        shaped = conv_out.view(-1, 4608)
        output = self.fc(shaped)
        return output
