
import os
import torch
import random
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import random
from tqdm import tqdm
from collections import deque
from models_fully import DiscardNet, PongNet, ChiNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tot = 8
action_mapping = {'摸牌': 0, '打牌': 1, '碰': 2, '吃': 3, '明杠': 4, '暗杠':4, '补杠':4, '和牌': 5, '补花后摸牌': 0, '杠后摸牌': 0}

order_mapping = dict([('W1', 0), ('W2', 0), ('W3', 0), ('W4', 0), ('W5', 0), ('W6', 0), ('W7', 0), ('W8', 0), ('W9', 0), ('B1', 0), ('B2', 0), ('B3', 0), ('B4', 0), ('B5', 0), ('B6', 0),
                          ('B7', 0), ('B8', 0), ('B9', 0), ('T1', 0), ('T2', 0), ('T3', 0), ('T4', 0), ('T5', 0), ('T6', 0), ('T7', 0), ('T8', 0), ('T9', 0), ('F1', 0), ('F2', 0), ('F3', 0),
                          ('F4', 0), ('J1', 0), ('J2', 0), ('J3', 0)])

valid_actions = ['摸牌', '打牌', '碰', '吃', '明杠', '暗杠', '补杠', '补花后摸牌', '杠后摸牌']
huapai = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8']

# 0 mopai in the form action-type, list
# 1 discard in the form action_type, list
# 2 pong in the form action_type, list, single, last_player
# 3 chow in the form action_type, list, single, last_player
# 4 kong in the form action_type, list
# 5 hu in the form action_type, list, single, last_player

# difficult : how to maintain the #appear variable?
# solution : using public.appear + self.hand


def weights_init(m):
    name = m.__class__.__name__
    if name.find('Conv1d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif name.find('Linear') != -1:
        init.kaiming_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.0)


def initial_order():
    for (i, key) in enumerate(order_mapping.keys()):
        order_mapping[key] = i

initial_order()


def FeatureExtractor(PlayerList, actions, winner):

    requires_tiles = [np.zeros(34) for _ in range(4)]
    if action_mapping[actions[1]] in [2, 3, 4] and actions[0] == winner:
        #print("this is nice")
        requires_tiles[int(winner)][order_mapping[actions[3]]] = 1

    features = [player.Encode2TokyoHot() for player in PlayerList]

    # 34 x 1
    #[hand, angang, pong, gang, chi, out]
    #[hand, an, pong, gang, chi, left, require]

    # public features for players

    public  = [feat[2:] for feat in features]
    temp = []
    for feat in public:
        temp += feat

    # private features for players
    private = [feat[0:2] for feat in features]

    out_feat = [np.array(feat[:-1]) for feat in features]
    sum_feat = [np.array(x + temp) for x in private]

    # left features for players
    left = [4 - x.sum(axis=0) for x in sum_feat]

    return [np.concatenate([x, np.expand_dims(y, axis=0), np.expand_dims(z, axis=0)], axis=0) for (x, y, z) in zip(out_feat, left, requires_tiles)]


def single_file_reader(f):
    file = []
    counter = 0
    for line in f:
        aha = line.strip('\n').split('\t')
        if 2 <= counter <= 5:
            aha[1] = aha[1].strip("[").strip("]").split(',')
            aha[1] = [x.strip('\'') for x in aha[1]]
        elif counter > 4:
            aha = line.strip('\n').split('\t')
            if aha[-1] == '':
                aha = aha[:-1]
            aha[2] = aha[2].strip("[").strip("]").split(',')
            aha[2] = [x.strip('\'') for x in aha[2]]
        if counter > 0:
            file.append(aha)
        counter += 1
    return file


class Player(object):
    def __init__(self, id, h):
        self.id = id

        self.hand = dict([('W1', 0), ('W2', 0), ('W3', 0), ('W4', 0), ('W5', 0), ('W6', 0), ('W7', 0), ('W8', 0), ('W9', 0), ('B1', 0), ('B2', 0), ('B3', 0), ('B4', 0), ('B5', 0), ('B6', 0),
                          ('B7', 0), ('B8', 0), ('B9', 0), ('T1', 0), ('T2', 0), ('T3', 0), ('T4', 0), ('T5', 0), ('T6', 0), ('T7', 0), ('T8', 0), ('T9', 0), ('F1', 0), ('F2', 0), ('F3', 0),
                          ('F4', 0), ('J1', 0), ('J2', 0), ('J3', 0)])

        for x in h:
            if x in huapai:
                continue
            self.hand[x] += 1

        # self pong, gang, chi and angang
        pong = dict([('W1', 0), ('W2', 0), ('W3', 0), ('W4', 0), ('W5', 0), ('W6', 0), ('W7', 0), ('W8', 0), ('W9', 0), ('B1', 0), ('B2', 0), ('B3', 0), ('B4', 0), ('B5', 0), ('B6', 0),
                          ('B7', 0), ('B8', 0), ('B9', 0), ('T1', 0), ('T2', 0), ('T3', 0), ('T4', 0), ('T5', 0), ('T6', 0), ('T7', 0), ('T8', 0), ('T9', 0), ('F1', 0), ('F2', 0), ('F3', 0),
                          ('F4', 0), ('J1', 0), ('J2', 0), ('J3', 0)])

        gang = dict([('W1', 0), ('W2', 0), ('W3', 0), ('W4', 0), ('W5', 0), ('W6', 0), ('W7', 0), ('W8', 0), ('W9', 0), ('B1', 0), ('B2', 0), ('B3', 0), ('B4', 0), ('B5', 0), ('B6', 0),
                          ('B7', 0), ('B8', 0), ('B9', 0), ('T1', 0), ('T2', 0), ('T3', 0), ('T4', 0), ('T5', 0), ('T6', 0), ('T7', 0), ('T8', 0), ('T9', 0), ('F1', 0), ('F2', 0), ('F3', 0),
                          ('F4', 0), ('J1', 0), ('J2', 0), ('J3', 0)])

        chi = dict([('W1', 0), ('W2', 0), ('W3', 0), ('W4', 0), ('W5', 0), ('W6', 0), ('W7', 0), ('W8', 0), ('W9', 0), ('B1', 0), ('B2', 0), ('B3', 0), ('B4', 0), ('B5', 0), ('B6', 0), ('B7', 0),
                         ('B8', 0), ('B9', 0), ('T1', 0), ('T2', 0), ('T3', 0), ('T4', 0), ('T5', 0), ('T6', 0), ('T7', 0), ('T8', 0), ('T9', 0), ('F1', 0), ('F2', 0), ('F3', 0), ('F4', 0), ('J1', 0),
                         ('J2', 0), ('J3', 0)])

        self.out = dict([('W1', 0), ('W2', 0), ('W3', 0), ('W4', 0), ('W5', 0), ('W6', 0), ('W7', 0), ('W8', 0), ('W9', 0), ('B1', 0), ('B2', 0), ('B3', 0), ('B4', 0), ('B5', 0), ('B6', 0), ('B7', 0),
                         ('B8', 0), ('B9', 0), ('T1', 0), ('T2', 0), ('T3', 0), ('T4', 0), ('T5', 0), ('T6', 0), ('T7', 0), ('T8', 0), ('T9', 0), ('F1', 0), ('F2', 0), ('F3', 0), ('F4', 0), ('J1', 0),
                         ('J2', 0), ('J3', 0)])

        self.angang = dict([('W1', 0), ('W2', 0), ('W3', 0), ('W4', 0), ('W5', 0), ('W6', 0), ('W7', 0), ('W8', 0), ('W9', 0), ('B1', 0), ('B2', 0), ('B3', 0), ('B4', 0), ('B5', 0), ('B6', 0), ('B7', 0),
                         ('B8', 0), ('B9', 0), ('T1', 0), ('T2', 0), ('T3', 0), ('T4', 0), ('T5', 0), ('T6', 0), ('T7', 0), ('T8', 0), ('T9', 0), ('F1', 0), ('F2', 0), ('F3', 0), ('F4', 0), ('J1', 0),
                         ('J2', 0), ('J3', 0)])

        self.info = [pong, chi, gang]

    def Encode2TokyoHot(self):
        # TODO : modified to values
        hand = [i for i in self.hand.values()]
        angang = list(self.angang.values())
        pong = [i for i in self.info[0].values()]
        chi = [i for i in self.info[1].values()]
        gang = [i for i in self.info[2].values()]
        out = [i for i in self.out.values()]
        return [hand, angang, pong, gang, chi, out]

    def append(self, actions):

        # ‘摸牌': 0, '打牌': 1, '碰': 2, '吃': 3, '杠‘: 4, '和牌': 5
        action_no = action_mapping[actions[1]]
        if action_no == 5:
            return

        if actions[1] == '暗杠':
            pid, action, tiles, ltile, lpid = actions
            if self.id == pid:
                for tile in tiles:
                    self.hand[tile] -= 1
                    self.angang[tile] += 1 
                return   

        if action_no == 0:
            pid, action, tiles = actions
            if self.id == pid:
                for tile in tiles:
                    self.hand[tile] += 1

        elif action_no == 1:
            pid, action, tiles = actions
            if self.id == pid:
                for tile in tiles:
                    self.hand[tile] -= 1
                    self.out[tile] += 1
            

        elif action_no == 2 or action_no == 3 or action_no == 4:
            pid, action, tiles, ltile, lpid = actions
            
            if self.id == pid:
                self.hand[ltile] += 1
                for tile in tiles:
                    self.hand[tile] -= 1
                    self.info[action_no - 2][tile] += 1
            elif self.id == lpid:
                self.out[ltile] -= 1

def canChi(handcards, card):
    card_no = order_mapping[card]
    cards = list(handcards.values())
                    # three cases (card, *, *), (*, card, *), (*, *, card)
    if card_no >= 0 and card_no < 9:
        if card_no >= 0 and card_no<=6 and cards[card_no+1]>0 and cards[card_no+2]>0:
            return True
        if card_no >= 1 and card_no<=7 and cards[card_no-1]>0 and cards[card_no+1]>0:
            return True
        if card_no >= 2 and card_no<=8 and cards[card_no-1]>0 and cards[card_no-2]>0:
            return True
    if card_no >= 9 and card_no < 18:
        if card_no >= 0+9 and card_no<=6+9 and cards[card_no+1]>0 and cards[card_no+2]>0:
            return True
        if card_no >= 1+9 and card_no<=7+9 and cards[card_no-1]>0 and cards[card_no+1]>0:
            return True
        if card_no >= 2+9 and card_no<=8+9 and cards[card_no-1]>0 and cards[card_no-2]>0:
            return True

    if card_no >= 18 and card_no < 27:
        if card_no >= 0+18 and card_no<=6+18 and cards[card_no+1]>0 and cards[card_no+2]>0:
            return True
        if card_no >= 1+18 and card_no<=7+18 and cards[card_no-1]>0 and cards[card_no+1]>0:
            return True
        if card_no >= 2+18 and card_no<=8+18 and cards[card_no-1]>0 and cards[card_no-2]>0:
            return True
    return False


all_channel = 7
hidden_channel = 64
card_num = 34
batch_size = 32
path_play = './output2017/PLAY'
path_mo = './output2017/MO'
dataset_all = [[], [], [], [], []]

class Agent(object):
    def __init__(self):
        self.memory = deque()

        # self.discard_agent = Discard().to(device)
        '''
        self.gang_agent = Gang().to(device)
        self.pong_agent = Pong().to(device)
        self.chi_agent = Chi().to(device)
        '''
        self.nets = [0, DiscardNet(all_channel*34).to(device), PongNet(all_channel*34).to(device), ChiNet(all_channel*34).to(device), PongNet(all_channel*34).to(device)]
        # self.PongList = [Gang().to(device) for i in range(3)]
        for i in range(1, 5):
            print("Yes! Using Kaiming init!")
            self.nets[i].apply(weights_init)

    def learn(self):
        pass

    def file_loader(self, file):

        dataset = [ ]
        players = file[1:5]
        actions = file[5:]

        # which is to maintain the public discard tiles
        out =  dict([('W1', 0), ('W2', 0), ('W3', 0), ('W4', 0), ('W5', 0), ('W6', 0), ('W7', 0), ('W8', 0), ('W9', 0), ('B1', 0), ('B2', 0), ('B3', 0), ('B4', 0), ('B5', 0), ('B6', 0), ('B7', 0),
                     ('B8', 0), ('B9', 0), ('T1', 0), ('T2', 0), ('T3', 0), ('T4', 0), ('T5', 0), ('T6', 0), ('T7', 0), ('T8', 0), ('T9', 0), ('F1', 0), ('F2', 0), ('F3', 0), ('F4', 0), ('J1', 0),
                     ('J2', 0), ('J3', 0)])

        #TODO : when not "和牌", quit
        hupai = (file[-1][1] == '和牌')
        if not hupai:
            return

        score = int(file[0][1])
        winner = file[-1][0]

        PlayerList = []
        for line in players:
            PlayerList.append(Player(line[0], line[1]))

        for index in range(len(actions)-1):
            line = actions[index]
            line_next = actions[index + 1]
            if line[1] not in valid_actions or ((line[1] in ['摸牌', '补花后摸牌', '杠后摸牌']) and (line[2][0] in huapai)):
                continue

            player_id = line[0]
            action_type = action_mapping[line[1]]
            # generate labels for actions
            if action_type == 1:
                label = order_mapping[line[2][0]]
            else:
                if action_type == 3:
                    for x in range(3):
                        if line[2][x] == str(line[3]):
                            label = x + 1
                            break
                else:
                    label = 1

            states = FeatureExtractor(PlayerList, line, winner)
            cur_id = line[0]
            ac_ty = line[1]
            if cur_id != winner and action_type == 1:
                card = line[2][0]
                # print('winint=', int(winner))
                # print('win=', winner)
                if PlayerList[int(winner)].hand[card] >=2 and line_next[1] != '碰' :
                    dataset_all[2].append((states[int(winner)], 0, score))

                if canChi(PlayerList[int(winner)].hand, card) and line_next[1] != '吃' and (int(cur_id) == (int(winner)+3)%4) :
                    dataset_all[3].append((states[int(winner)], 0, score))

                if PlayerList[int(winner)].hand[card] >=2 and line_next[1] != '杠':
                    dataset_all[4].append((states[int(winner)], 0, score))

            # modified this line, and you can read all players
            if player_id == winner and 1 <= action_type <= 4:
                dataset_all[action_type].append((states[int(player_id)], label, score))

            for player in PlayerList:
                player.append(line)

    def dataloader(self):
        files = os.listdir(path_play)
        cnt = 0
        for file in files:
            if cnt> 200000:
                break
            if cnt % 1000 == 0:
                print(cnt)

            cnt += 1
            # print(file)
            position = path_play + '/' + file
            with open(position, "r", encoding='utf-8') as f:
                file = single_file_reader(f)
            self.file_loader(file)
        
        files = os.listdir(path_mo)
        for file in files:
            if cnt>270000:
                break
            if cnt % 1000 == 0:
                print(cnt)
            cnt += 1
            # print(file)
            position = path_mo + '/' + file
            with open(position, "r", encoding='utf-8') as f:
                file = single_file_reader(f)
            self.file_loader(file)


def train_test_split(dataset, train_rate=0.85, valid_rate=0.15):
    n_total = len(dataset)
    n_train, n_valid = int(train_rate * n_total), int(valid_rate * n_total)
    idx = list(range(n_total))

    idx_mask = np.ones(n_total)
    train_idx = random.sample(idx, n_train)
    idx_mask[train_idx] = 0

    rest_idx = np.nonzero(idx_mask)[0]
    valid_idx = random.sample(list(rest_idx), n_valid)
    idx_mask[valid_idx] = 0

    test_idx = np.nonzero(idx_mask)[0]
    
    train_data = [dataset[i] for i in train_idx]
    valid_data = [dataset[i] for i in valid_idx]
    test_data = [dataset[i] for i in test_idx]

    return train_data, valid_data, test_data

def myDataLoader(dataset, batch_size=256):
    n_data = len(dataset)
    n_batch = n_data//batch_size
    if n_data % batch_size != 0:
        n_batch += 1

    idx_mask = np.ones(n_data)
    for i in range(n_batch):
        num = min(batch_size, n_data - i*batch_size)
        index = list(np.nonzero(idx_mask)[0])
        sampled = random.sample(index, num)
        idx_mask[sampled] = 0
        
        temp = [dataset[j] for j in sampled]
        yield temp

# announce data_train, data_valid, data_test, agent
'''
train_loader = myDataLoader(train_data, batch_size=256)
valid_loader = myDataLoader(valid_data, batch_size=256)
test_loader = myDataLoader(test_data, batch_size=256)
'''

def train_net(train_net, dataset, net_number, epoch=3000):
    #train_net.load_state_dict(torch.load("models1/checkpoint_79_0.39933474900586413.pth"))
    train_net.train()
    train_data, valid_data, test_data = train_test_split(dataset)
    n_train  = len(train_data)

    optimizer = optim.Adam(train_net.parameters(), lr = 1e-3, weight_decay=1e-5)
    train_loader = myDataLoader(train_data, batch_size=64)

    for i in tqdm(range(epoch)):
        count_train = 0
        count_zero = 0
        batch_size = 128
        for batch_xy in myDataLoader(train_data, batch_size=batch_size):
            
            feats = [x[0] for x in batch_xy]
            labels = [x[1]for x in batch_xy]
            weights = [x[2] for x in batch_xy]

            feats = torch.FloatTensor(np.stack(feats, axis=0)).to(device)
            labels = torch.LongTensor(labels).to(device)
            weights = torch.FloatTensor(weights).to(device)

            optimizer.zero_grad()

            '''
            log_prob = torch.log(predict)
            shape = predict.shape
            
            logits = -1 * log_prob[np.arange(shape[0]), labels]
            loss = (logits * weights).sum()
            '''

            predict = train_net(feats.view(-1, all_channel*34))
            # loss_fn = nn.CrossEntropyLoss().to(device)
            loss = F.cross_entropy(predict, labels).to(device)
            count_zero += labels.sum().item()
            #print("the pre is {}  lab is {}".format(predict[1],labels[1]))

            loss.backward()
            optimizer.step()

            count_train += torch.eq(torch.argmax(predict, dim=1), labels).sum().item()

        print("In epoch {} : train accuracy {} zero {}".format(i, count_train/n_train, 1 - count_zero/n_train))

        # TODO : validation accuracy
        if (i+1) % 20 == 0 and i > 0:
            test_net(train_net, valid_data, i, net_number)
            # torch.save(train_net.state_dict(),'models{}/checkpoint_{}_{}.pth'.format(net_number,i, count_train/n_train))


def train_discard(train_net, dataset, net_number, epoch=3000):
    train_net.load_state_dict(torch.load("models1/checkpoint_39_0.5721991605730202_1e-3.pth"))
    train_net.train()
    train_data, valid_data, test_data = train_test_split(dataset)
    n_train  = len(train_data)

    optimizer = optim.Adam(train_net.parameters(), lr = 1e-3, weight_decay=1e-5)
    train_loader = myDataLoader(train_data, batch_size=64)

    for i in tqdm(range(epoch)):
        count_train = 0
        count_zero = 0
        batch_size = 128
        for batch_xy in myDataLoader(train_data, batch_size=batch_size):
            
            feats = [x[0] for x in batch_xy]
            labels = [x[1]for x in batch_xy]
            weights = [x[2] for x in batch_xy]

            feats = torch.FloatTensor(np.stack(feats, axis=0)).to(device)
            labels = torch.LongTensor(labels).to(device)
            weights = torch.FloatTensor(weights).to(device)

            optimizer.zero_grad()

            pred_logits = train_net(feats.view(-1, all_channel*34))
            prob_mask = torch.index_select(feats, 1, torch.LongTensor([0]).to(device)).view(-1, 34)
            masked_predict = torch.where(prob_mask > 0, pred_logits, torch.full_like(pred_logits, -1e20).to(device))

            predict = F.softmax(masked_predict, dim=1)
            '''
            log_prob = torch.log(predict)
            shape = predict.shape
            
            logits = -1 * log_prob[np.arange(shape[0]), labels]
            loss = (logits * weights).sum()
            '''

            loss = F.cross_entropy(predict, labels).to(device)
            count_zero += labels.sum().item()
            #print("the pre is {}  lab is {}".format(predict[1],labels[1]))

            loss.backward()
            optimizer.step()

            count_train += torch.eq(torch.argmax(predict, dim=1), labels).sum().item()

        print("In epoch {} : train accuracy {} zero {}".format(i, count_train/n_train, 1 - count_zero/n_train))

        # TODO : validation accuracy
        if (i+1) % 20 == 0 and i > 0:
            test_discard(train_net, valid_data, i, net_number)
            # torch.save(train_net.state_dict(),'models{}/checkpoint_{}_{}.pth'.format(net_number,i, count_train/n_train))


def test_net(net, test_dataset,i, net_number):
    # print(test_dataset)
    
    count = 0
    n_test = len(test_dataset)
    test_loader = myDataLoader(test_dataset)

    net.eval()
    for batch_xy in test_loader:
        feats = [x[0] for x in batch_xy]
        labels = [x[1] for x in batch_xy]

        feats = torch.FloatTensor(np.stack(feats, axis=0)).to(device)
        labels = torch.LongTensor(labels).to(device)

        predict = net(feats.view(-1, all_channel*34))

        pred_labels = torch.argmax(predict, dim=1)
        # print(predict.shape)
        # print(labels)
        count += torch.eq(pred_labels, labels).sum().item()
    
    if n_test != 0:
        print("The test accuracy is {}".format(count/n_test))
        torch.save(net.state_dict(),'models{}/checkpoint_{}_{}_1e-3.pth'.format(net_number,i, count/n_test))
    return count/n_test


def test_discard(net, test_dataset,i, net_number):
    # print(test_dataset)
    
    count = 0
    n_test = len(test_dataset)
    test_loader = myDataLoader(test_dataset)

    net.eval()
    for batch_xy in test_loader:
        feats = [x[0] for x in batch_xy]
        labels = [x[1] for x in batch_xy]

        feats = torch.FloatTensor(np.stack(feats, axis=0)).to(device)
        labels = torch.LongTensor(labels).to(device)

        pred_logits = net(feats.view(-1, all_channel*34))
        prob_mask = torch.index_select(feats, 1, torch.LongTensor([0]).to(device)).view(-1, 34)
        masked_predict = torch.where(prob_mask > 0, pred_logits, torch.full_like(pred_logits, -1e20).to(device))
        predict = F.softmax(masked_predict, dim=1)

        pred_labels = torch.argmax(predict, dim=1)
        # print(predict.shape)
        # print(labels)
        count += torch.eq(pred_labels, labels).sum().item()
    
    if n_test != 0:
        print("The test accuracy is {}".format(count/n_test))
        torch.save(net.state_dict(),'models{}/checkpoint_{}_{}_1e-3.pth'.format(net_number,i, count/n_test))
    return count/n_test
            

if __name__ == "__main__":
    myagent = Agent()
    '''
    with open('train.txt', "r", encoding='utf-8') as f:
        file  = single_file_reader(f)
    
    dataset = myagent.file_loader(file)
    print(dataset)
    '''
    myagent.dataloader()
    train_discard(myagent.nets[1], dataset_all[1],1)
    #train_net(myagent.nets[4], dataset_all[4],4)
        
        
    

