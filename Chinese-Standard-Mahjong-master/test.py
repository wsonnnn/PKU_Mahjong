import json
import numpy as np
class t(object):
    def __init__(self):
        self.num = 9
        self.list = [1,2,3,4,5,6,7]
        self.dict = {1:1,2:2,3:3}
        
f = t()
class MyEncoder(json.JSONEncoder):
    def default(self,obj):
        #convert object to a dict
        d={}
        d['__class__']=obj.__class__.__name__
        d['__module__']=obj.__module__
        d.update(obj.__dict__)
        return d
 
class MyDecoder(json.JSONDecoder):
    def __init__(self):
        json.JSONDecoder.__init__(self,object_hook=self.dict2object)
    def dict2object(self,d):
        #convert dict to object
        if '__class__' in d:
            class_name=d.pop('__class__')
            module_name=d.pop('__module__')
            module=__import__(module_name)
            class_=getattr(module,class_name)
            args=dict((key.encode('ascii'), value) for key, value in d.items())
            inst = class_(**args)#create new instance
        else:
            inst=d
        return inst

