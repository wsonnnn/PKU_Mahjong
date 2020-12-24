import numpy as np
import json
import sys
#import random
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

def get_tile(tile):
    hand = np.zeros(35, dtype = 'int64')
    for index in range(5,18):
        hand[str_index(tile[index])] += 1
    return hand

def L_P():
    print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
    sys.stdout.flush()
total_request = []
def get_input(hang, y):
    ss = []
    for i in range(hang):
        k = input()
        if len(k)>1:
            ss.append(k)
    return ss

if __name__ == '__main__':
    Input = get_input(3, 0)
    #total_request.append(Input)
    _, ID, quan = Input[-1].split(" ")
    print("PASS")
    L_P()
    Input = get_input(2, 0)
    #total_request.append(Input)
    first_card = Input[-1].split(" ")
    hand = get_tile(first_card)
    print("PASS")
    L_P()
    turnID = 1
    #hang = 5
    while(True):
        Input = get_input(1,1)
        #total_request.append(Input)
        turnID += 1
        Input = Input[-1].split(" ")
        if Input[0] == '2':
            hand[str_index(Input[1])] += 1
            for i in range(35):
                if hand[i] != 0:
                    print("PLAY",index_str(i))
                    hand[i] -= 1
                    break
            L_P()
            continue
        else:
            print("PASS")
            L_P()
            continue