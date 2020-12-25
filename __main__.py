import numpy as np
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
#import random
from MahjongGB import MahjongFanCalculator
from SL_fully import agent
from copy import deepcopy
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

request = []
response = []
hand = np.zeros(34,dtype = 'int64')
TILE_NUM = 21
table_transform={'W':-1,'B':8,'T':17,'F':26,'J':30}
pack_type = {'PENG':0,"GANG":1,"CHI":2}
str_trans_action = {"PASS":0, "PLAY":1, "GANG":2, "BUGANG":3, "PENG":4, "CHI":5, "HU":6}
action_trans = {0:"PASS", 1:"PLAY", 2:"GANG", 3:"BUGANG", 4:"PENG", 5:"CHI", 6:"HU"}
#是否要加一个player目前pack番数存放
#是否要将两种gang分开编码
def get_range(card_type):
    if card_type<=8:
        return 0,8
    elif card_type <= 17:
        return 9,17
    elif card_type <= 26:
        return 18,26
    else:
        return False, False
#将牌型字符串进行编码
def index_str(index):
    if index <= 8:
        return "W"+str(index+1)
    elif index <= 17:
        return "B"+str(index-8)
    elif index <= 26:
        return "T"+str(index-17)
    elif index <= 30:
        return "F"+str(index-26)
    else:
        return "J"+str(index-30)
class Pack(object):
    #由于算番器兼容性，没有把offer换成trainid
    def __init__(self,type,tile,offer):
        self.type = type
        self.tile = tile # -1 0-34   -1 for an'gang
        self.offer = offer # 0 1 2 3 
def tuple_pack(pack):
    a = []
    a.append(pack.type)
    a.append(pack.tile)
    a.append(pack.offer)
    return tuple(a)
def tuple_hand(hand):
    h_l = []
    for i in range(34):
        for j in range(hand[i]):
            h_l.append(index_str(i))
    return tuple(h_l)

class PlayerData(object):
    def __init__(self, hand, my_id):
        #当前各玩家的手牌总数
        self.hand_num_list = np.ones(4, dtype = int) * 13
        #当前各个玩家的牌墙数量
        self.TileWallList = np.ones(4,dtype = int) * TILE_NUM
        #我当前的手牌
        self.hand = hand
        #我当前的id
        self.id = my_id
        #当前已用牌堆
        self.used_card = np.zeros(34, dtype = 'int64')
        #当前每个player已打出牌，按照顺序而非总体
        #牌型编码 万：0-8，饼 9-17 条 18-26 风27-30 中发白 31-33  
        self.used = {0:[],1:[],2:[],3:[]}
        self.pack_list = {0:[], 1:[], 2:[], 3:[]}

    def get_myid(self):
        return self.id

    def init_input(self, card_type, chupai = False):
        #[hand, an, pong, gang, chi, left, require]
        angang = np.zeros(34)
        gang = np.zeros(34)
        peng = np.zeros(34)
        chi = np.zeros(34)
        card = np.zeros(34)
        if not chupai:
            card[card_type] +=1
        used = self.used_card + self.hand
        for pack in self.pack_list[0]:
            if pack.type == "PENG":
                peng[str_index(pack.tile)] += 1
            elif pack.type == "CHI":
                chi[str_index(pack.tile)] += 1
            else:
                if pack.offer == 0:
                    angang[str_index(pack.tile)] +=1
                else:
                    gang[str_index(pack.tile)] +=1

        features = torch.FloatTensor([self.hand, angang, peng, gang, chi, 4-used, card]).to(device)
        return features

    #为了方便训练，需要将playID相对自己编号，因为针对上家和下家出牌的需要的操作是有区别的
    def get_trainid(self, id):
        return (id - self.id + 4) % 4
    def check_bugang(self, card_type):#这里补杠只能在当前抽到补杠牌时才能补杠，可能之后再补杠的策略更好
        avail_type = []
        next_id = (self.id + 1) % 4
        #按照裁判代码，检查了自己和下家的牌墙余数
        if self.TileWallList[next_id] == 0 or self.TileWallList[self.id]:
            return False, avail_type
        len_pack = len(self.pack_list[0])
        for i in range(len_pack):
            type_id = str_index(self.pack_list[0][i].tile)
            if self.pack_list[0][i].type == "PENG" and type_id  == card_type:
                avail_type.append(card_type)
            if self.pack_list[0][i].type == "PENG" and self.hand[type_id] != 0:
                avail_type.append(type_id)
        if len(avail_type)!=0:
            return True, avail_type
        else:
            return False, avail_type

    def check_gang(self, my_card, card_type):
        avail_type = []
        if my_card:
            for i in range(34):
                if self.hand[i] == 4:
                    avail_type.append(i)
        elif self.hand[card_type] == 3:
            avail_type.append(card_type)
        if len(avail_type) == 0:
            return False, avail_type
        else:
            return True, avail_type

    def check_hu(self, my_card, card_type, pid, gang, meng, quan):
        hand = self.hand
        pack_list = []
        #hand[card_type] += 1
        have_peng = False
        for i in range(len(self.pack_list[0])):
            pack_list.append(tuple_pack(self.pack_list[0][i]))
            if self.pack_list[0][i].type == "PENG" and self.pack_list[0][i].tile == index_str(card_type):
                have_peng = True
        tuple_p = tuple(pack_list)
        tuple_h = tuple_hand(self.hand)
        next_id = (pid + 1) % 4
        is_last = False
        is_jue = False
        if my_card and self.used_card[card_type] == 3:
            is_jue = True
        if (not my_card) and self.used_card[card_type] == 4:
            is_jue = True
        if not my_card and self.TileWallList[next_id] == 0:
            is_last = True
        if my_card and self.TileWallList[1] == 0:
            is_last = True
        if not my_card and gang:
            is_gang = True
        if my_card and self.hand[card_type] == 3:
            is_gang = True
        if have_peng and my_card:
            is_gang = True
        reward = 0
        try:
            ans = MahjongFanCalculator(tuple_p, tuple_h, index_str(card_type), 1, my_card, is_jue, is_gang, is_last, meng, quan)
        except Exception as err:
            is_hu = False
        else:
            is_hu = True
            for i in range(len(ans)):
                reward += ans[i][0]
        return is_hu, reward

    def check_peng(self, card_type):
        avail_type = []
        if self.hand[card_type] == 2 or self.hand[card_type] == 3:
            avail_type.append(card_type)
        if len(avail_type) == 0:
            return False, avail_type
        else:
            return True, avail_type

    def check_chi(self, card_type, train_id):
        avail_type = []
        if train_id != 3:
            return False, avail_type
        a, b = get_range(card_type)
        if a==False:
            return False, avail_type
        if card_type-2>=a and self.hand[card_type-2]!=0 and self.hand[card_type-1]!=0:
            avail_type.append(3)
        if card_type-1>=a and card_type+1<=b and self.hand[card_type-1]!=0 and self.hand[card_type+1] !=0:
            avail_type.append(2)
        if card_type + 2 <= b and self.hand[card_type+2]!=0 and self.hand[card_type+1]!=0:
                avail_type.append(1)
        if len(avail_type) != 0:
            return True, avail_type
        else: return False, avail_type



    def get_peng_use(self, id, chupai, peng_type, penged_id):
        #得出peng的pack
        pack = Pack("PENG", index_str(peng_type), penged_id)
        #加入到id对应的pack list
        self.pack_list[id].append(pack)
        #对于出牌进行维护
        self.chupai(id,chupai)
        #对于碰牌对于手牌数进行维护，因为出牌已经将手牌数-1，所以这里只用-2
        self.hand_num_list[id] -= 2
        self.used_card[peng_type] += 2
        if id == 0:
            self.hand[peng_type] -= 2
    
    def get_chi_card(self, chi_card, chi_type):
        if chi_card == chi_type:
            return chi_type - 1, chi_type + 1
        elif chi_card == chi_type - 1:
            return chi_type, chi_type + 1
        else: 
            return chi_type - 1, chi_type
    
    def get_chi_use(self, id, chi_type, chied_id, card_type, chi_card):
        if card_type < chi_card:
            num_last = 2
        elif card_type == chi_card:
            num_last = 1
        else:
            num_last = 0
        
        pack = Pack("CHI", index_str(chi_type), num_last)
        self.pack_list[id].append(pack)
        self.chupai(id, card_type)
        self.hand_num_list[id] -= 2
        a, b = self.get_chi_card(chi_card, chi_type)
        self.used_card[a] += 1
        self.used_card[b] += 1
        if id == 0:
            self.hand[a] -= 1
            self.hand[b] -= 1
    
    def get_angang(self, id, gang_num, playid):
        self.hand_num_list[id] -= 4
        if id == 0:
            pack = Pack("GANG", index_str(gang_num), playid)
            self.pack_list[id].append(pack)
            self.used_card[gang_num] += 4
            self.hand[gang_num] -= 4
        else:
            pack = Pack(pack_type["GANG"], -1, id)
            self.pack_list[id].append(pack)
            
    
    def get_gang(self, id, gang_type, past_id):
        pack = Pack("GANG", index_str(gang_type), past_id)
        self.pack_list[id].append(pack)
        self.hand_num_list[id] -= 3
        self.used_card[gang_type] += 3
        if id == 0:
            self.hand[gang_type] -=3
    
    def get_bugang(self, id, gang_type, play_id):
        flag = 1
        for i in range(len(self.pack_list[id])):
            if str_index(self.pack_list[id][i].tile) == gang_type and self.pack_list[id][i].type == "PENG":
                self.pack_list[id].pop(i)
                flag = 0
                break
        if flag:
            print("BUG FOUND TRY BUGANG BUT PENG NOT FOUND!")
            return
        pack = Pack("GANG", index_str(gang_type), play_id)
        self.pack_list[id].append(pack)
        self.hand_num_list[id] -= 1
        self.used_card[gang_type] += 1
        if id == 0:
            self.hand[gang_type] -= 1
        
    def chupai(self, id, card_type):
        if id == 0:
            self.hand[card_type] -= 1
        self.used[id].append(card_type)
        self.used_card[card_type] += 1
        self.hand_num_list[id] -= 1
    #对于手牌和牌墙数量进行维护，如果是自己抽牌，需要card_type,否则无所谓，可以传-1
    def get_newtile(self, card_type, id):
        if card_type != -1:
            self.hand[card_type] += 1
            self.TileWallList[id] -= 1
            self.hand_num_list[id] += 1
        else:
            self.TileWallList[id] -= 1
            self.hand_num_list[id] += 1

    
def str_index(str):
    return table_transform[str[0]]+int(str[1])

#return turnID data request response state
#turnID int 当前turn数
#data 上一个回合自己上传的data
#request 当前输入
#response agent response history
def get_tile(tile):
    hand = np.zeros(34, dtype = 'int64')
    for index in range(5,18):
        hand[str_index(tile[index])] += 1
    return hand

#action:
#PASS
#HU
#GANG                           (用上一一回合打出的牌，所以其他player知道你杠了什么)
#PLAY CARD_PLAY
#GANG CARD_GANG                 (用自己的牌，其他player不会知道你到底杠了什么，CARD_GANG是给裁判的，其他人不知道)
#BUGANG CARD_BUGANG             
#PENG CARD_PENG
#CHI CARD_MID CARD_PLAY
def print_hand(hand):
    hhh = []
    for i in range(34):
        for j in range(hand[i]):
            hhh.append(index_str(i))
    return hhh
    
def L_P():
    print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
    sys.stdout.flush()

def get_action(data):
    return 1,2,3

def past_processing(input):
    Input = input.split(" ")
    id = int(Input[1])
    card_type = str_index(Input[-1])
    return id, card_type

total_request = []

def get_input(hang):
    ss = []
    for i in range(hang):
        k = input()
        if len(k)>1:
            ss.append(k)
    return ss
def longterm_programming():
    Input = get_input(3)[-1]
    total_request.append(Input)
    _, ID, quan = Input.split(" ")
    meng = int(ID)
    quan = int(quan)
    print("PASS")
    L_P()
    Input = get_input(2)[-1]
    total_request.append(Input)
    first_card = Input.split(" ")
    hand = get_tile(first_card)
    my_data = PlayerData(hand,int(ID))
    print("PASS")
    L_P()
    #check_action_flag = 0
    turnID = 1
    gang_num = -1
    agent = agent()
    # TODO : add models
    # 1 : 打牌， 2:碰， 3:吃，4:杠
    while(True):
        Input = get_input(1)[-1]
        total_request.append(Input)
        turnID += 1
        Input = Input.split(" ")
        #if check_action_flag:#TODO
        #    pass
        avail_type = []
        avail_num = []
        if Input[0] == '2':
            #print(print_hand(my_data.hand))
            card_type = str_index(Input[1])
            #my_data.get_newtile(card_type, 0)
            bu_gang, action1 = my_data.check_bugang(card_type)
            gang, action2 = my_data.check_gang(my_card = True, card_type = -1)
            hu, action3 = my_data.check_hu(my_card = True, card_type = card_type, pid = 0, gang = False, meng= meng, quan=quan)
            if bu_gang:
                avail_type.append(str_trans_action["BUGANG"])
                avail_num.append(action1)
            if gang:
                avail_type.append(str_trans_action["GANG"])
                avail_num.append(action2)
            if hu:
                avail_type.append(str_trans_action["HU"])
                avail_num.append(action3)
            my_data.get_newtile(card_type, 0)
            avail_type.append(str_trans_action["PLAY"])
            avail_num.append(my_data.hand)
            if bu_gang:
                print("BUGANG",index_str(action1[0]))
            elif gang:
                print("GANG",index_str(action2[0]))
            elif hu:
                print("HU")
            #action, x, y = get_action(my_data, avail_type, avail_num)
            #if action_trans[action] == "GANG":
            #    gang_num = x
            #print(action_trans[action],index_str(x))
            else:
                #TODO
                # 打牌， 碰， 吃， 杠
                features = my_data.init_input('W0', chupai=True)
                mask = features[0]
                pred_logits = agent.nets[1](features.view(1, -1))
                masked_logits = torch.where(mask > 0, pred_logits, torch.fully_like(pred_logits, -1e20))
                # pred_prob = F.softmax(masked_logits, dim=1)
                
                result = torch.argmax(masked_logits, dim=1)[0].item()
                print("PLAY", index_str(result))

                '''
                for i in range(34):
                    if my_data.hand[i] != 0:
                        print("PLAY", index_str(i))
                        break
                '''
            L_P()
            continue
        elif Input[0] == '3':
            playid = int(Input[1])
            train_id = my_data.get_trainid(playid)
            if Input[2] == "DRAW":
                my_data.get_newtile(-1, train_id)
                print("PASS")
                L_P()
                continue
            elif Input[2] == "PLAY":
                card_type = str_index(Input[3])
                my_data.chupai(train_id, card_type)
                if train_id == 0:#这是我自己干的
                    print("PASS")
                else:#这是别人出的牌
                    peng, action1 = my_data.check_peng(card_type)
                    gang, action2 = my_data.check_gang(my_card = False, card_type = card_type)
                    hu, action3 = my_data.check_hu(my_card = False, card_type = card_type, pid = train_id, gang = False, meng=meng, quan=quan)
                    chi, action4 = my_data.check_chi(card_type, train_id)
                    if peng:
                        avail_type.append(str_trans_action["PENG"])
                        avail_num.append(action1)
                    if gang:
                        avail_type.append(str_trans_action["GANG"])
                        avail_num.append(action2)
                    if hu:
                        avail_type.append(str_trans_action["HU"])
                        avail_num.append(action3)
                    if chi:
                        avail_type.append(str_trans_action["CHI"])
                        avail_num.append(action4)
                    avail_type.append(str_trans_action["PASS"])
                    avail_num.append([])
                    if hu:
                        print("HU")
                    else:
                        max_logit = 0
                        max_action = 0
                        action_tuple = 0
                        raw_features = my_data.init_input(card_type, chupai=False)
                        features = deepcopy(raw_features).view(1, -1)
                        if peng + chi + gang == 0:
                            print("PASS")
                        else:
                            pred_peng = agent.nets[2](features) * peng # 2
                            max_peng = torch.argmax(pred_peng, dim=1)[0].item()
                            pred_chi = agent.nets[3](features) * chi # 4
                            pred_recons = torch.cat([pred_chi[0][0].view(1, 1), (pred_chi[0][1] + pred_chi[0][2] + pred_chi[0][3]).view(1, 1)], dim=1)
                            max_chi = torch.argmax(pred_recons, dim=1)[0].item()
                            pred_gang = agent.nets[4](features) * gang # 2
                            max_gang = torch.argmax(pred_gang, dim=1)[0].item()

                            if max_peng + max_chi + max_gang == 0:
                                print("PASS")
                            else:
                                if pred_gang[0][1] >= pred_peng[0][1] and pred_gang[0][1]>= (pred_chi[0][1]+pred_chi[0][2]+pred_chi[0][3])*max_chi:
                                    print("GANG {}".format(index_str(card_type)))
                                elif pred_peng[0][1] >= pred_gang[0][1] and pred_peng[0][1]>= (pred_chi[0][1]+pred_chi[0][2]+pred_chi[0][3])*max_chi:
                                    features = my_data.init_input(card_type, chupai=True)
                                    features[0][card_type] -= 2
                                    features[2][card_type] += 3

                                    mask = features[0]
                                    pred_logits = agent.nets[1](features.view(1, -1))
                                    masked_logits = torch.where(mask > 0, pred_logits, torch.fully_like(pred_logits, -1e20))
                                    # pred_prob = F.softmax(masked_logits, dim=1)
                
                                    result = torch.argmax(masked_logits, dim=1)[0].item()
                                    print("PENG {}".format(str_index(result)))
                                else:
                                    pred_recons = pred_chi[[0, 0, 0], [1, 2, 3]]
                                    chi_mask = [0,0,0]
                                    for x in avail_type:
                                        chi_mask[x-1] = 1
                                    chi_mask = torch.FloatTensor(chi_mask).view(1, 3).to(device)
                                    
                                    masked_recons = pred_recons * chi_mask
                                    chi_res = torch.argmax(masked_recons, dim=1)[0].item()

                                    middle = card_type + 1 - chi_res

                                    features = my_data.init_input(card_type, chupai=True)

                                    features[0][card_type] += 1
                                    for i in range(3):
                                        features[0][card_type - chi_res + i] -= 1
                                        features[4][carfd_type - chi_res + i] += 1

                                    mask = features[0]
                                    pred_logits = agent.nets[1](features.view(1, -1))
                                    masked_logits = torch.where(mask > 0, pred_logits, torch.fully_like(pred_logits, -1e20))

                                    discard_result = torch.argmax(masked_logits, dim=1)[0].item()
                                    print("CHI {} {}".format(str_index(middle), str_index(discard_result)))

                L_P()
                continue

            elif Input[2] == "PENG":
                card_type = str_index(Input[3])
                past_input = total_request[turnID-1]
                id, peng_type = past_processing(past_input)
                #past_train_id = my_data.get_trainid(id)
                my_data.get_peng_use(train_id, card_type, peng_type, id)
                if train_id == 0:
                    print("PASS")
                else:
                    peng, action1 = my_data.check_peng(card_type)
                    gang, action2 = my_data.check_gang(my_card = False, card_type = card_type)
                    hu, action3 = my_data.check_hu(my_card = False, card_type = card_type, pid = train_id, gang = False, meng=meng, quan=quan)
                    chi, action4 = my_data.check_chi(card_type, train_id)
                    if peng:
                        avail_type.append(str_trans_action["PENG"])
                        avail_num.append(action1)
                    if gang:
                        avail_type.append(str_trans_action["GANG"])
                        avail_num.append(action2)
                    if hu:
                        avail_type.append(str_trans_action["HU"])
                        avail_num.append(action3)
                    if chi:
                        avail_type.append(str_trans_action["CHI"])
                        avail_num.append(action4)
                    avail_type.append(str_trans_action["PASS"])
                    avail_num.append([])
                    if hu:
                        print("HU")
                    else:
                        max_logit = 0
                        max_action = 0
                        action_tuple = 0
                        raw_features = my_data.init_input(card_type, chupai=False)
                        features = deepcopy(raw_features).view(1, -1)
                        if peng + chi + gang == 0:
                            print("PASS")
                        else:
                            pred_peng = agent.nets[2](features) * peng # 2
                            max_peng = torch.argmax(pred_peng, dim=1)[0].item()
                            pred_chi = agent.nets[3](features) * chi # 4
                            pred_recons = torch.cat([pred_chi[0][0].view(1, 1), (pred_chi[0][1] + pred_chi[0][2] + pred_chi[0][3]).view(1, 1)], dim=1)
                            max_chi = torch.argmax(pred_recons, dim=1)[0].item()
                            pred_gang = agent.nets[4](features) * gang # 2
                            max_gang = torch.argmax(pred_gang, dim=1)[0].item()

                            if max_peng + max_chi + max_gang == 0:
                                print("PASS")
                            else:
                                if pred_gang[0][1] >= pred_peng[0][1] and pred_gang[0][1]>= (pred_chi[0][1]+pred_chi[0][2]+pred_chi[0][3])*max_chi:
                                    print("GANG {}".format(index_str(card_type)))
                                elif pred_peng[0][1] >= pred_gang[0][1] and pred_peng[0][1]>= (pred_chi[0][1]+pred_chi[0][2]+pred_chi[0][3])*max_chi:
                                    features = my_data.init_input(card_type, chupai=True)
                                    features[0][card_type] -= 2
                                    features[2][card_type] += 3

                                    mask = features[0]
                                    pred_logits = agent.nets[1](features.view(1, -1))
                                    masked_logits = torch.where(mask > 0, pred_logits, torch.fully_like(pred_logits, -1e20))
                                    # pred_prob = F.softmax(masked_logits, dim=1)
                
                                    result = torch.argmax(masked_logits, dim=1)[0].item()
                                    print("PENG {}".format(str_index(result)))
                                else:
                                    pred_recons = pred_chi[[0, 0, 0], [1, 2, 3]]
                                    chi_mask = [0,0,0]
                                    for x in avail_type:
                                        chi_mask[x-1] = 1
                                    chi_mask = torch.FloatTensor(chi_mask).view(1, 3).to(device)
                                    
                                    masked_recons = pred_recons * chi_mask
                                    chi_res = torch.argmax(masked_recons, dim=1)[0].item()

                                    middle = card_type + 1 - chi_res

                                    features = my_data.init_input(card_type, chupai=True)

                                    features[0][card_type] += 1
                                    for i in range(3):
                                        feature[0][card_type - chi_res + i] -= 1
                                        feature[4][card_type - chi_res + i] += 1

                                    mask = features[0]
                                    pred_logits = agent.nets[1](features.view(1, -1))
                                    masked_logits = torch.where(mask > 0, pred_logits, torch.fully_like(pred_logits, -1e20))

                                    discard_result = torch.argmax(masked_logits, dim=1)[0].item()
                                    print("CHI {} {}".format(str_index(middle), str_index(discard_result)))
                L_P()
                continue
            elif Input[2] == "CHI":
                chi_type = str_index(Input[3])
                card_type = str_index(Input[4])
                past_input = total_request[turnID-1]
                pid, chi_card = past_processing(past_input)
                past_train_id = my_data.get_trainid(pid)
                my_data.get_chi_use(train_id, chi_type, past_train_id, card_type, chi_card)
                if train_id == 0:
                    print("PASS")
                else:
                    peng, action1 = my_data.check_peng(card_type)
                    gang, action2 = my_data.check_gang(my_card = False, card_type = card_type)
                    hu, action3 = my_data.check_hu(my_card = False, card_type = card_type, pid = train_id, gang = False, meng=meng, quan=quan)
                    chi, action4 = my_data.check_chi(card_type, train_id)
                    if peng:
                        avail_type.append(str_trans_action["PENG"])
                        avail_num.append(action1)
                    if gang:
                        avail_type.append(str_trans_action["GANG"])
                        avail_num.append(action2)
                    if hu:
                        avail_type.append(str_trans_action["HU"])
                        avail_num.append(action3)
                    if chi:
                        avail_type.append(str_trans_action["CHI"])
                        avail_num.append(action4)
                    avail_type.append(str_trans_action["PASS"])
                    avail_num.append([])
                    if hu:
                        print("HU")
                    else:
                        max_logit = 0
                        max_action = 0
                        action_tuple = 0
                        raw_features = my_data.init_input(card_type, chupai=False)
                        features = deepcopy(raw_features).view(1, -1)
                        if peng + chi + gang == 0:
                            print("PASS")
                        else:
                            pred_peng = agent.nets[2](features) * peng # 2
                            max_peng = torch.argmax(pred_peng, dim=1)[0].item()
                            pred_chi = agent.nets[3](features) * chi # 4
                            pred_recons = torch.cat([pred_chi[0][0].view(1, 1), (pred_chi[0][1] + pred_chi[0][2] + pred_chi[0][3]).view(1, 1)], dim=1)
                            max_chi = torch.argmax(pred_recons, dim=1)[0].item()
                            pred_gang = agent.nets[4](features) * gang # 2
                            max_gang = torch.argmax(pred_gang, dim=1)[0].item()

                            if max_peng + max_chi + max_gang == 0:
                                print("PASS")
                            else:
                                if pred_gang[0][1] >= pred_peng[0][1] and pred_gang[0][1]>= (pred_chi[0][1]+pred_chi[0][2]+pred_chi[0][3])*max_chi:
                                    print("GANG {}".format(index_str(card_type)))
                                elif pred_peng[0][1] >= pred_gang[0][1] and pred_peng[0][1]>= (pred_chi[0][1]+pred_chi[0][2]+pred_chi[0][3])*max_chi:
                                    features = my_data.init_input(card_type, chupai=True)
                                    features[0][card_type] -= 2
                                    features[2][card_type] += 3

                                    mask = features[0]
                                    pred_logits = agent.nets[1](features.view(1, -1))
                                    masked_logits = torch.where(mask > 0, pred_logits, torch.fully_like(pred_logits, -1e20))
                                    # pred_prob = F.softmax(masked_logits, dim=1)
                
                                    result = torch.argmax(masked_logits, dim=1)[0].item()
                                    print("PENG {}".format(str_index(result)))
                                else:
                                    pred_recons = pred_chi[[0, 0, 0], [1, 2, 3]]
                                    chi_mask = [0,0,0]
                                    for x in avail_type:
                                        chi_mask[x-1] = 1
                                    chi_mask = torch.FloatTensor(chi_mask).view(1, 3).to(device)
                                    
                                    masked_recons = pred_recons * chi_mask
                                    chi_res = torch.argmax(masked_recons, dim=1)[0].item()

                                    middle = card_type + 1 - chi_res

                                    features = my_data.init_input(card_type, chupai=True)

                                    features[0][card_type] += 1
                                    for i in range(3):
                                        feature[0][card_type - chi_res + i] -= 1
                                        feature[4][card_type - chi_res + i] += 1

                                    mask = features[0]
                                    pred_logits = agent.nets[1](features.view(1, -1))
                                    masked_logits = torch.where(mask > 0, pred_logits, torch.fully_like(pred_logits, -1e20))

                                    discard_result = torch.argmax(masked_logits, dim=1)[0].item()
                                    print("CHI {} {}".format(str_index(middle), str_index(discard_result)))
                L_P()
                continue
            elif Input[2] == "GANG":
                past_input = total_request[turnID-1].split(" ")
                if past_input[2] == "DRAW":#暗杠
                    my_data.get_angang(train_id, gang_num, playid)
                    gang_num = -1
                else:
                    gang_card = str_index(past_input[-1])
                    past_id = int(past_input[1])
                    my_data.get_gang(train_id, gang_card, past_id)
                print("PASS")
                L_P()
                continue
            elif Input[2] == "BUGANG":
                card_type = str_index(Input[3])
                is_hu, action = my_data.check_hu(my_card = False, card_type = card_type, pid = train_id, gang = True,meng= meng,quan= quan)
                my_data.get_bugang(train_id, card_type, playid)
                if is_hu:
                    print("HU")
                else:
                    print("PASS")
                L_P()
                continue
            else:
                print("error")
        

if __name__ == '__main__':
    longterm_programming()
