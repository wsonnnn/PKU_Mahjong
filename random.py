import numpy as np
import json
import random
request = []
response = []
hand = np.zeros(35,dtype = 'int64')
public = np
TILE_NUM = 21
table_transform={'W':-1,'B':8,'T':17,'F':26,'J':30}
class Pack(object):
    def __init__(self,type,tile,offer):
        self.type = type
        self.tile = tile
        self.offer = offer
class PlayerData(object):
    def __init__(self, t_num, hand, p_list, my_id):
        self.Pack_list = p_list
        self.TileWallNumber = t_num
        self.hand = hand
        self.id = my_id
        #当前已用牌堆
        self.used_card = np.zeros(35, dtype = 'int64')
        #当前每个player已打出牌，按照顺序而非总体
        self.used_id = {0:[],1:[],2:[],3:[]}
    def get_myid(self):
        return self.id
    def get_peng_use(self,id,)
    def chupai(self, id, card_type):
        self.used_id[id].append(card_type)
        self.used_card[card_type] += 1
    def get_newtile(card_type):
        self.hand[card_type] += 1
    
def str_index(str):
    return table_transform[str[0]]+int(str[1])

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
def getrand():
    set = []
    for i in range(35):
        for j in range(hand[i]):
            set.append(i)
    return random.sample(set,1)[0]

#return turnID data request response state
#turnID int 当前turn数
#data 上一个回合自己上传的data
#request 当前输入
#response agent response history

#为了方便训练，需要将playID相对自己编号，因为针对上家和下家出牌的需要的操作是有区别的
def get_train_ID(myid, ID):


def get_input():
    Input = input()
    Input = json.loads(Input)
    turnID = len(Input['responses'])
    data = None
    if 'data' in Input:
        data = Input['data']
    request = Input['requests']
    response = Input['responses']
    turnID = len(response)
    MyRequest = []
    MyResponse = []
    for i in range(turnID):
        k = request[i].split(" ")
        j = response[i].split(" ")
        MyRequest.append(k)
        MyResponse.append(j)
    MyRequest.append(request[turnID].split(" "))
    return turnID+1, data, MyRequest, MyResponse
def get_tile(tile):
    hand = np.zeros(35, dtype = 'int64')
    for index in range(5,18):
        hand[str_index(tile[index])] += 1
    return hand
def reget(data, request, response, turnID):
    if data == None:
        if turnID == 1:
            return None
        else:
            my_hand = get_tile(request[1])
            my_id = str(request[0][1])
            return PlayerData(TILE_NUM, my_hand, [], my_id)
    my_data = (PlayerData)data[0]
    #这回合我自己进行抽牌
    if request[-1][0] == '2':
        my_data.get_newtile(str_index(request[-1][1]))
    #这回合其他玩家进行操作
    if request[-1][0] == '3':
        playID = int(request[-1][1])
        trainID = get_train_ID(my_data.get_myid(), playID)
        if request[-1][2] == "PLAY":
            card = request[-1][3]
            card = str_index(card)
            my_data.chupai(playID, card)
        if request[-1][2] == "PENG":
            #取得上上回合出的牌
            peng_card = str_index(request[-2][-1])

            


def sample():
    turnID, data, request, response = get_input()
    itmp, ID, quan = request[0]
    my_card = reget(data, request, response)
    response_dict = {}
    if(turnID < 2):
        response_dict['response']="PASS"
    else:
        for i in range(turnID):
            k = Input['requests'][i].split(" ")
            j = Input['responses'][i].split(" ")
            request.append(k)
            response.append(j)
        request.append(Input['requests'][turnID].split(" "))
        itmp,ID,quan = request[0]#string
        #print(request)
        #print(response)
        for index in range(5,18):
            hand[str_index(request[1][index])]+=1
        for index in range(2,turnID+1):
            if request[index][0] == '2':
                hand[str_index(request[index][1])]+=1
        for index in range(0,turnID):
            if response[index][0]=='PLAY':
                hand[str_index(response[index][1])]-=1
        if request[-1][0] == '2':
            k = getrand()
            k = index_str(k)
            response_json['response']="PLAY "+k
        else:
            response_json['response']="PASS"
    response_json = json.dumps(response_json)
    print(response_json)
        
    #print(Input['requests'])
#sample()
if __name__ == '__main__':
    sample()
