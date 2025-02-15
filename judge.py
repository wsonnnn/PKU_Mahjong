import numpy as np
import json
import random
from copy import deepcopy
from MahjongGB import MahjongFanCalculator
import sys
class tileWall(object):
    def __init__(self, wall):
        self.wall = wall
    def get_init_hand(self):
        hand = self.wall[0:13]
        self.wall = self.wall[13:]
    def get_next(self):
        a = self.wall[0]
        self.wall = self.wall[1:]
        return a
class Judge(object):
    def __init__(self, same_tile = False):
        self.agent = []
        self.run_time_policy = same_tile
        self.tileWall_Save = []
        self.tileWall = []
    def get_newdata(self):
        List = np.arange(136)
        sampleList = np.random.permutation(List)
        if self.run_time_policy:
            if len(self.tileWall_Save) == 0:
                print("get")
                self.tileWall_Save = sampleList[:34]
                d = np.setdiff1d(sampleList, self.tileWall_Save)
                d = np.random.permutation(d)
                a = d[0:34]
                b = d[34:68]
                c = d[68:]
            else:
                d = np.setdiff1d(sampleList, self.tileWall_Save)
                d = np.random.permutation(d)
                a = d[:34]
                b = d[34:68]
                c = d[68:]
            return self.tileWall_Save, a, b, c
        else:
            return sampleList[0:34], sampleList[34:68], sampleList[68:102], sampleList[102:]
    def get_newWall(self):
        walls = deepcopy(self.get_newdata())
        #print(walls)
        for p in walls:
            for i in range(len(p)):
                p[i] /= 4
        return walls
    def next_t(self, t):
        return (t+1) % 4
    def pick_card(self, id):
        k = -1
        if len(self.tileWall[i]) != 0:
            k = self.tileWall[i][0]
            self.tileWall[i] = self.tileWall[1:]
        return k
    def get_agent(self,agent):
        self.agent.append(agent)
    def prepare_game(self, train_mode = 0):
        self.tileWall = self.get_newWall()
        if len(self.agent) != 4:
            print("agent not fulled")
            return
        self.currentPlayer = random.randint(0,3)
        for i in range(4):
            self.agent[i].get_init_tile(self.tileWall[i][0:13])
            self.tileWall[i] = self.tileWall[i][13:]
        new_card = self.pick_card(self.currentPlayer)
        while(True):
            for i in range(4):
                state, throwCard = agent.
    def chef_play(self, ai):
        now_turn = random.randint(0,3)
        Wall = j.get_newWall()
        wall = []
        ID = {}
        for i in range(4):
            wall.append(tileWall(Wall[i]))
        for i in range(4):
            ai[i].get_init(now_turn, wall[now_turn].get_init_hand())
            ID[i] = now_turn
            now_turn = next_t(now_turn)
        while(True):
            k = ai[now_turn].get_action("DRAW",wall[now_turn].get_next())
            action = []
            for i in range(4):
                action.append(ai[i].get_action(now_turn, k))
            


        

j = Judge(same_tile = True)
for i in range(2):
    a,b,c,d = j.get_newWall()



