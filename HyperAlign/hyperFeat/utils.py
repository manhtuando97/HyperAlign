from time import time
import logging,inspect
import pickle
from itertools import islice
import os.path


dir_f = ''

folder_pickles = "pickles/"

def returnPathhf2vec():
    return dir_f

def isPickle(fname):

    return os.path.isfile('pickles/'+fname+'.pickle')

def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def partition(lst, n):
    division = len(lst) / float(n)

    lst = list(lst)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def restoreVariableFromDisk(name):
    t0 = time()
    val = None
    with open(folder_pickles + name + '.pickle', 'rb') as handle:
        val = pickle.load(handle)
    t1 = time()

    return val

def satinary_check(args, n_embd1, n_embd2):
    n_nodes = min(n_embd1.shape[0], n_embd2.shape[0])
    n_v = max(n_embd1.shape[0], n_embd2.shape[0])

    import math
    if args.epd > n_nodes:
        args.epd = math.floor(0.5 * n_nodes * n_nodes / n_v)
    elif math.floor(0.5 * n_nodes) > args.epd:
        args.epd = math.floor(math.sqrt(0.1) * n_nodes)
    else:
        args.epd = math.floor(math.sqrt(0.2) * n_nodes)

    return n_embd1, (n_embd2, args.epd)

def saveVariableOnDisk(f,name):
    t0 = time()

    with open(folder_pickles + name + '.pickle', 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time()

    return
