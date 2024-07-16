# -*- coding: utf-8 -*-
from time import time
import logging,inspect
import pickle
from itertools import islice
import os.path

#dir_f = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dir_f = ''
#folder_pickles = dir_f+ "/pickles/"
folder_pickles = "pickles/"

def returnPathhf2vec():
    return dir_f

def isPickle(fname):
    #return os.path.isfile(dir_f+'/../pickles/'+fname+'.pickle')
    return os.path.isfile('pickles/'+fname+'.pickle')

def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def partition(lst, n):
    division = len(lst) / float(n)
    #print(type(lst))
    lst = list(lst)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def restoreVariableFromDisk(name):
    t0 = time()
    val = None
    with open(folder_pickles + name + '.pickle', 'rb') as handle:
        val = pickle.load(handle)
    t1 = time()

    return val

def saveVariableOnDisk(f,name):
    t0 = time()
    #print(folder_pickles)
    with open(folder_pickles + name + '.pickle', 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time()

    return





