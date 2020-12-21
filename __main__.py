import numpy as np
import json
#import random
from MahjongGB import MahjongFanCalculator
import sys
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
        self.hand_num_list = np.ones(4, dtype = int) * 13
        self.TileWallList = np.ones(4,dtype = int) * TILE_NUM
        self.hand = hand
        self.id = my_id
        #当前已用牌堆
        self.used_card = np.zeros(34, dtype = 'int64')
        #当前每个player已打出牌，按照顺序而非总体
        self.used = {0:[],1:[],2:[],3:[]}
        self.pack_list = {0:[], 1:[], 2:[], 3:[]}

    def get_myid(self):
        return self.id
    
    #为了方便训练，需要将playID相对自己编号，因为针对上家和下家出牌的需要的操作是有区别的
    def get_trainid(self, id):
        return (id - self.id + 4) % 4
    def check_bugang(self, card_type):#这里补杠只能在当前抽到补杠牌时才能补杠，可能之后再补杠的策略更好
        avail_type = []
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

    def check_hu(self, my_card, card_type, pid, gang):
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
            ans = MahjongFanCalculator(tuple_p, tuple_h, index_str(card_type), 1, my_card, is_jue, is_gang, is_last, 0, 0)
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
            avail_type.append(card_type-1)
        if card_type-1>=a and card_type+1<=b and self.hand[card_type-1]!=0 and self.hand[card_type+1] !=0:
            avail_type.append(card_type)
        if card_type + 2 <= b and self.hand[card_type+2]!=0 and self.hand[card_type+1]!=0:
                avail_type.append(card_type+1)
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
            hu, action3 = my_data.check_hu(my_card = True, card_type = card_type, pid = 0, gang = False)
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
                for i in range(34):
                    if my_data.hand[i] != 0:
                        print("PLAY", index_str(i))
                        break
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
                    hu, action3 = my_data.check_hu(my_card = False, card_type = card_type, pid = train_id, gang = False)
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
                    elif peng:
                        print("PENG",end = " ")
                        for i in range(34):
                            if hand[i] != 0:
                                print(index_str(i))
                                break
                    elif gang:
                        print("GANG",index_str(action2[0]))
                    elif chi:
                        print("CHI",index_str(action4[0]),end = " ")
                        for i in range(34):
                            if hand[i] != 0:
                                print(index_str(i))
                                break
                    else:
                        print("PASS")
                    #avail_num.append(my_data.hand)
                    #if my_data.check_peng(card_type):
                    #    avail_action.append(str_trans_action["PENG"])
                    #if my_data.check_hu(my_card = False, card_type = card_type):
                    #    avail_action.append(str_trans_action["HU"])
                    #if my_data.check_chi(card_type, train_id):
                    #    avail_action.append(str_trans_action["CHI"])
                    #if my_data.check_gang(my_card = False, card_type = card_type):
                    #    avail_action.append(str_trans_action["GANG"])
                    #avail_action.append(str_trans_action["PASS"])
                    #action, x, y = get_action(my_data, avail_type, avail_num)
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
                    hu, action3 = my_data.check_hu(my_card = False, card_type = card_type, pid = train_id, gang = False)
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
                    elif peng:
                        print("PENG",end = " ")
                        for i in range(34):
                            if hand[i] != 0:
                                print(index_str(i))
                                break
                    elif gang:
                        print("GANG",index_str(action2[0]))
                    elif chi:
                        print("CHI",index_str(action4[0]),end = " ")
                        for i in range(34):
                            if hand[i] != 0:
                                print(index_str(i))
                                break
                    else: print("PASS")
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
                    hu, action3 = my_data.check_hu(my_card = False, card_type = card_type, pid = train_id, gang = False)
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
                    elif peng:
                        print("PENG",end = " ")
                        for i in range(34):
                            if hand[i] != 0:
                                print(index_str(i))
                                break
                    elif gang:
                        print("GANG",index_str(action2[0]))
                    elif chi:
                        print("CHI",index_str(action4[0]),end = " ")
                        for i in range(34):
                            if hand[i] != 0:
                                print(index_str(i))
                                break
                    else: print("PASS")
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
                is_hu, action = my_data.check_hu(my_card = False, card_type = card_type, pid = train_id, gang = True)
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
