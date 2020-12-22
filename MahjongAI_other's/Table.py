#-*- coding: utf-8 -*-　
import BasicDefinition
import numpy as np 
from six.moves import cPickle
from Agent import Agent
from BasicDefinition import CardIndex,WindIndex
import random
from SimpleAgent import RandomAgent,OneStepAgent
from evalScore import evalScore 
import time
class Table:
    MAX_Agent = 4
    currentAgent = 0
    throwsAndCombination = [] ## 紀錄[已經丟的牌,已翻開的組合,當局產生的組合]
    loseReason = [[],[],[],[]]
    def __init__(self,saved=False):
        self.autoSaved = saved
        self.agents = []
        self.deck = []

    def newGame(self) :
        self.deckInitial()
        self.throwedCards=[[],[],[],[]]
        self.scoreBoard = [0]*4
        for agent in self.agents:
            agent.reset()
    
    def gameEnd(self,save = False, player = None,pickle_name = None): 
        for i in range(4):
            self.agents[i].gameEnd(self.win,self.lose,self.scoreBoard[i])
        if save:
            self.agents[player].save(pickle_name)
    
    def addAgent(self,newAgent):
        if len(self.agents) < self.MAX_Agent :
            self.agents.append(newAgent)
        else :
            print ('Agent reach the maximum agent number!')
    
    def deal(self) :
        self.shuffleDeck()
        if len(self.agents) != self.MAX_Agent:
            print ('Current agent number : ', len(self.agents))
            print ('No enough agents')
        else :
            for agent in self.agents:
                handCard = self.deck[0:13]
                self.deck = self.deck[13::]
                agent.initialHandCard(handCard)

    def gameStart(self,verbose = False,UI=False,pause=False):
        self.currentAgent = random.randint(0,3)
        for i in range(self.MAX_Agent):
            wind = WindIndex[i]
            agent = self.agents[(i+self.currentAgent)%self.MAX_Agent]
            agent.setWind(wind)
        
        ###
        if verbose: 
            print ('Game start !')
            print ('Agent ',self.currentAgent,' is first!')
            print ('-------------------------------------')
        ###

        newCard = self.pickCard()
        while True:
            table = None
            ###
            if verbose :
                
                print ('current handcards for every agent :')
                for agent in self.agents :
                    agent.printHandCard()
                
                print ("\nAgent ",self.currentAgent,"'s action")
            if (verbose or UI) and newCard != None:
                    print ('get ',CardIndex[newCard]) 
            if UI:
                table = self.__getVisibleTable()
                self.__addToken(table,self.currentAgent)
                for row in table:
                    print ('|'.join(map(str,row)))
                time.sleep(0.1)
            ###

            agent  = self.agents[self.currentAgent]
            state ,throwCard = agent.takeAction(newCard,verbose)
                                    
                            
            if state == '自摸' : 
                if verbose :
                    print ('Agent ',self.currentAgent,':',state,end='')
                    self.__printCards(throwCard)
                cards = throwCard
                winAgent = self.currentAgent
                break
            

            ###
            if verbose or UI: 
                print ('Throw ',CardIndex[throwCard])
            if UI:
                time.sleep(0.1)
            ###

            assert throwCard < 34 and throwCard >= 0,('the card you throw is ',throwCard)
            ## find who is the next
            nextAgent = (self.currentAgent+1) % self.MAX_Agent
            for i in range(self.MAX_Agent):
                if i != self.currentAgent:
                    agent = self.agents[i]
                    info = agent.check(self.currentAgent,throwCard)
                    tmpCards = info[0]#[1 1 1]
                    tmpState = info[1]#'吃'、'碰'、'槓'
                    assert throwCard == info[2]#丟出來的那張
                    
                    ###
                    if verbose : 
                        print ('Agent ',i,':',tmpState,end='')
                        self.__printCards(tmpCards)
                        #input()
                    ###
                    assert self.__cardChecker(tmpCards,tmpState) 
                    if tmpState == '過':
                        continue
                    
                    ## 胡 > 碰槓 > 吃
                    if tmpState == '胡' :
                        nextAgent = i
                        cards = tmpCards
                        state = tmpState
                        if self.autoSaved :
                            self.loseReason[self.currentAgent].append([cards,throwCard])
                        break
                    if tmpState == '碰':
                        assert state != '槓' and state != '碰'
                        state = tmpState
                        cards = tmpCards
                        nextAgent = i
                    elif tmpState == '槓':
                        assert state != '碰' and state != '槓'
                        state = tmpState
                        cards = tmpCards
                        nextAgent = i
                    elif tmpState == '吃' and (state != '碰' or state != '槓'):
                        state = tmpState
                        cards = tmpCards
                        nextAgent = i
                    else :
                        print ('No define state \'',tmpState,"\'")
                        sys.exit()
            
            
            if state == '胡':
                winAgent = nextAgent
                loseAgent = self.currentAgent
                break
			            
            if state == '吃' or state == '碰' or state == '槓':
                if verbose or UI:
                    print ('Agent ',nextAgent,' get ',CardIndex[throwCard])
                if UI:
                    time.sleep(0.1)
                takeAgent = nextAgent            
                takeCards = cards                

                if state == '槓':
                    newCard = self.pickCard()
                    ## if deck is empty it means no winner in this round
                    if newCard == None :
                        state = '流局'
                        break
                else:
                    #newCard = throwCard
                    newCard = None

            else :
                self.throwedCards[self.currentAgent].append(throwCard)
                if verbose or UI:
                    print ('No agnet get ',CardIndex[throwCard])
                if UI: 
                    time.sleep(0.1)
                takeAgent = None
                takeCards = None
                newCard = self.pickCard()
                ## if deck is empty it means no winner in this round
                if newCard == None :
                    state = '流局'
                    break

            ## broadcast information
            for i in range(self.MAX_Agent):
                self.agents[i].update(self.currentAgent,takeAgent,takeCards,throwCard,verbose)
            self.currentAgent = nextAgent
            if verbose :
                print ('-------------------------------------')
        if state == '胡' :
            if verbose:
                print ('贏家 : ',winAgent)
                print ('放槍 : ',loseAgent)
            score = evalScore(cards[0]+cards[1],cards[0],cards[1],winAgent,self.agents[winAgent].wind)
            self.scoreBoard[winAgent] += score * 3
            if score <= 25:
                for i in range(4):
                    if i != winAgent :
                        self.scoreBoard[i] -=score
            else :
                self.scoreBoard[loseAgent] -= (score*3 -50)
                for i in range(4):
                    if i != winAgent and i != loseAgent:
                        self.scoreBoard[i] -= 25
            self.win = winAgent
            self.lose = loseAgent
            
            return winAgent,loseAgent,self.scoreBoard
        elif state == '自摸':
            if verbose:
                print (winAgent,'自摸')
            score = evalScore(cards[0]+cards[1],cards[0],cards[1],winAgent,self.agents[winAgent].wind)
            self.scoreBoard[winAgent] += score * 3
            for i in range(4):
                if i != winAgent :
                    self.scoreBoard[i] -=score
            
            self.win = winAgent
            self.lose = None

            return winAgent,None,self.scoreBoard
        elif state == '流局' :
            if verbose:
                print (state)
            
            self.win = None
            self.lose= None
            self.scoreBoard = [None]*4
            
            return None,None,None
        else:
            assert 0==1

    def pickCard(self):
        if len(self.deck) > 14:
            return self.deck.pop()
        else:
            return None
            
    def deckInitial(self):
        self.deck = []
        for i in range(34):
            if i < 34 :
                self.deck.append([i]*4)    
            else :
                self.deck.append([i])
        self.deck = sum(self.deck,[])

    def shuffleDeck(self):
        random.shuffle(self.deck)    

    def __cardChecker(self,cards,state):
        if state != '胡':
            assert all([(card < 34 and card >= 0) for card in cards])
        error = True
        if len(cards)==0 and state == '過':
            error=False
        elif len(cards) == 4 and len(set(cards)) == 1 and state == '槓':
            error = False
        elif len(cards) == 3 and len(set(cards)) == 1 and state == '碰':
            error = False
        elif len(cards) == 3 and len(set(cards)) == 3:
            if all([(card > 0 and card < 10) or (card>10 and card < 20) or (card>20 and card<30) for card in cards]):
                cards = sorted(cards)
                if cards[0]+1 == cards[1] and cards[1]+1 == cards[2] :
                    error=False
        elif state=='胡' and len(cards)==2:
            cards =cards[0]+cards[1]
            error = False
            for card in cards:
                if len(card)==2 and card[0]==card[1]:
                    pass
                elif len(card)==3:
                    cardLen = len(set(card))
                    if cardLen==3 and card[0]+1==card[1] and card[1]+1==card[2]:
                        pass
                    elif cardLen == 1:
                        pass
                    else:
                        error=True
                        break
                elif len(card)==4 and len(set(card))==1:
                    pass
                else :
                    error=True
                    break
        else :
            print (cards)
            print (state)
            assert 0==1
        if error :
            print ('state : ',state,', cards : ',cards)
            print ('card Checkerror')

        return not error

    def __getVisibleTable(self):
        visibleTable=[]
        visibleTable.append(['*'*69])
        for i in range(self.MAX_Agent):
            cards = ''
            for card in self.throwedCards[i]:
                cards += CardIndex[card]+','
            visibleTable.append(['Agent '+str(i)+' : ' +cards])
        
        visibleTable.append(['*'*69])
        visibleTable.append([' '*69])
        visibleTable.append(['      ','-'*55,'      '])
        ### get agent 1's cards on board 
        cards = self.agents[1].getCardsOnBoard()
        cards = sum(cards,[])
        r1 = ['      ']
        r2 = ['      ']
        for card in cards :
            chinese = CardIndex[card]
            r1.append(chinese[0])
            if len(chinese) == 2:
                r2.append(chinese[1])
            else :
                r2.append('  ')
        if len(cards)!=0:
            r1.append('')
            r2.append('')
        visibleTable.append(r1)
        visibleTable.append(r2)
        visibleTable.append('')
        ### get agent 2's cards on board
        cards = self.agents[2].getCardsOnBoard()
        cards = sum(cards,[])
        for i in range(16) :
            r = [' ']
            if i < len(cards) :
                card = CardIndex[cards[i]]
                if len(card) == 1:
                    r.append(card.center(3))
                else:
                    r.append(card)
                r.append(' '*55)
            else :
                r.append(' '*60)
            visibleTable.append(r)
        ### get agent 0's cards on board
        
        offset = len(visibleTable)-16
        cards = self.agents[0].getCardsOnBoard()
        cards = sum(cards,[])
        for i in range(16) :
            if i < len(cards) :
                card = CardIndex[cards[i]]
                if len(card) == 1:
                    visibleTable[i+offset].append(card.center(3))
                else:
                    visibleTable[i+offset].append(card)

            else :
                visibleTable[i+offset][-1] += ' '*5 
            visibleTable[i+offset].append('')
        ### get agent 3's cards on board
        cards = self.agents[3].getCardsOnBoard()
        cards = sum(cards,[])
        r1 = ['      ']
        r2 = ['      ']
        for card in cards :
            chinese = CardIndex[card]
            if len(chinese) == 2:
                r1.append(chinese[0])
                r2.append(chinese[1])
            else :
                r1.append('  ')
                r2.append(chinese[0])
        if len(cards)!=0:
            r1.append('')
            r2.append('')
        visibleTable.append(r1)
        visibleTable.append(r2)
        visibleTable.append(['      ','-'*55,'      '])
        ## get agent 3's hand cards
        handcard = sorted(self.agents[3].handcard[:])
        r1 = ['      ']
        r2 = ['      ']
        for card in handcard:
            chinese = CardIndex[card]
            if len(chinese) == 2:
                r1.append(chinese[0])
                r2.append(chinese[1])
            else :
                r1.append('  ')
                r2.append(chinese[0])

        visibleTable.append(r1)
        visibleTable.append(r2)
        visibleTable.append(['      ','-'*55,'      '])
        visibleTable.append([' '*69])


        return visibleTable

    def __addToken(self,table,token):
        length = len(table)
        if token == 0:
            table[length-15][-1]='o'
        elif token == 1:
            space = int(len(table[0][0])/2)
            table[length-28][0]=' '*space+'o'+' '*space
        elif token == 2:
            table[length-15][0]='o'
        elif token == 3:
            space = int(len(table[0][0])/2)
            table[-1][0]=' '*space+'o'+' '*space
    def __printCards(self,cards):
        print (self.__cards2Chinese(cards))

    def __cards2Chinese(self,Cards):
        cards = []
        for element in Cards:
            if type(element) == type(list()):
                cards.append(self.__cards2Chinese(element))
            else:
                cards.append(CardIndex[element])
        return cards
