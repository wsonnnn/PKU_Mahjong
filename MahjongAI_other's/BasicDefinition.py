#-*- coding: utf-8 -*-　
CardIndex = {   '白'  : 0, '中'  : 10, '發'  : 20,
                '一萬': 1, '一條': 11, '一餅': 21,
                '二萬': 2, '二條': 12, '二餅': 22,
                '三萬': 3, '三條': 13, '三餅': 23,
                '四萬': 4, '四條': 14, '四餅': 24,
                '五萬': 5, '五條': 15, '五餅': 25,
                '六萬': 6, '六條': 16, '六餅': 26,
                '七萬': 7, '七條': 17, '七餅': 27,
                '八萬': 8, '八條': 18, '八餅': 28,
                '九萬': 9, '九條': 19, '九餅': 29,

                '東':30,'西':31,'南':32,'北':33,
                '春':34,'夏':35,'秋':36,'冬':37,
                '梅':38,'蘭':39,'竹':40,'菊':41,
                
                0:'白'  , 10:'中'  , 20:'發'  ,
                1:'一萬', 11:'一條', 21:'一餅',
                2:'二萬', 12:'二條', 22:'二餅',
                3:'三萬', 13:'三條', 23:'三餅',
                4:'四萬', 14:'四條', 24:'四餅',
                5:'五萬', 15:'五條', 25:'五餅',
                6:'六萬', 16:'六條', 26:'六餅',
                7:'七萬', 17:'七條', 27:'七餅',
                8:'八萬', 18:'八條', 28:'八餅',
                9:'九萬', 19:'九條', 29:'九餅',

                30:'東', 31:'西', 32:'南', 33:'北',
                34:'春', 35:'夏', 36:'秋', 37:'冬',
                35:'梅', 39:'蘭', 40:'竹', 41:'菊',
                }
WindIndex = {   '東風':0,'南風':1,'西風':2,'北風':3,
                0:'東風',1:'南風',2:'西風',3:'北風',
            }
PlayerAction = {'Throw':0,  0:'Throw',
                'Pass' :1,  1:'Pass' ,
                '吃'   :2,  2:'吃',
                '碰'   :3,  3:'碰',
                '槓'   :4,  4:'槓'
                }

if __name__=='__main__':
    print (len(CardIndex))
