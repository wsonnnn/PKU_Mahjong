import torch
import torch.nn as nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    
    def __init__(self, input_channel, out_channel):

        super(ResBlock, self).__init__()

        self.net = nn.Sequential(nn.Conv1d(input_channel, out_channel, 3, 1, 1),
                                 nn.BatchNorm1d(out_channel),
                                 nn.ReLU(),
                                 nn.Conv1d(out_channel, out_channel, 3, 1, 1))
    
    def forward(self, x):

        # TODO : add batch norm?
        out = F.relu(x + self.net(x))
        return out

class MahjongNet(nn.Module):

    def __init__(self, input, out, hidden=256):

        super(MahjongNet, self).__init__()
        '''
        self.embedding = nn.Conv1d(input_channel, hidden_channel, 3, 1, 1)
        resblocks = [ResBlock(hidden_channel, hidden_channel) for i in range(layers)]
        self.resblocks = nn.Sequential(*resblocks)
        '''
        self.fc1 = torch.nn.Linear(input,hidden)
        self.fc2 = torch.nn.Linear(hidden,hidden)
        self.fc3 = torch.nn.Linear(hidden,hidden)
        self.fc4 = torch.nn.Linear(hidden,hidden)
        self.fc5 = torch.nn.Linear(hidden,out)
        self.bn1 = nn.BatchNorm1d(input)
        self.bn2 = nn.BatchNorm1d(hidden)
    def forward(self, x):
        x = self.bn1(x)
        x = nn.ReLU()(self.fc1(x))
        x = F.dropout(x,p=0.3,training=self.training)
        x = nn.ReLU()(self.fc2(self.bn2(x)))
        x = F.dropout(x,p=0.3,training=self.training)
        x = nn.ReLU()(self.fc3(self.bn2(x)))
        x = F.dropout(x,p=0.3,training=self.training)
        x = nn.ReLU()(self.fc4(self.bn2(x)))
        x = F.dropout(x,p=0.3,training=self.training)
        x = nn.ReLU()(self.fc5(self.bn2(x)))
        return x

class DiscardNet(nn.Module):

    def __init__(self, input, hidden=64):

        super(DiscardNet, self).__init__()
        self.Mahjong = MahjongNet(input, 64, hidden=hidden)
        self.policy = nn.Linear(64, 34)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.Mahjong(x)
        out = nn.ReLU()(self.policy(out).view(-1, 34))
        out = self.softmax(out)
        return out

# for Chow, Pong, Kong etc.
class PongNet(nn.Module):

    def __init__(self, input):

        super(PongNet, self).__init__()
        self.Mahjong = MahjongNet(input, 256, hidden=256)
        self.policy = nn.Sequential(nn.Linear(256, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 2),
                                    nn.ReLU(),
                                    nn.Softmax(dim=1))
    
    def forward(self, x):

        size = x.shape
        feature = self.Mahjong(x)
        out = self.policy(feature)
        return out

class ChiNet(nn.Module):

    def __init__(self, input):

        super(ChiNet, self).__init__()
        self.Mahjong = MahjongNet(input, 256, hidden=256)
        self.policy = nn.Sequential(nn.Linear(256, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 4),
                                    nn.ReLU(),
                                    nn.Softmax(dim=1))
    
    def forward(self, x):
        size = x.shape
        feature = self.Mahjong(x)
        out = self.policy(feature)
        return out

if __name__ == "__main__":
    '''
    net = DiscardNet(23)
    x = torch.randn(1, 23, 34)
    print(net(x))
    print(net(x).shape)
    '''