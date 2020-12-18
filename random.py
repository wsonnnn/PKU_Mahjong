import numpy as np
import json
import random
request = []
response = []
hand = np.zeros(35,dtype = 'int64')
table_transform={'W':-1,'B':8,'T':17,'F':26,'J':30}
def str_index(str):
    return table_transform[str[0]]+int(str[1])
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
def sample():
    s = input()
    response_json = {}
    Input = json.loads(s)
    #Input = json.loads(Input)
    #print(Input)
    turnID = len(Input['responses'])
    #print(turnID)
    if(turnID < 2):
        response_json['response']="PASS"
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
