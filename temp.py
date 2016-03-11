import random
import numpy as np


itvec = list(range((250000 + 50000 - 1) // 50000))
print itvec
random.shuffle(itvec)
print itvec

"""

ids = [1, 23, 34, 89, 90, 215, 412, 560]
specs = [8333, 5644, 98, 345, 2984, 77, 43, 1090]

print "ids iniciais:"
print ids
print
print
print "specs iniciais:"
print specs
print
print

idxnotcold = [1, 2, 3, 5, 7]
print "indices nao cold: ", idxnotcold

ids = np.array(ids)
idsnotcold = ids[idxnotcold]
print "ids nao cold: ", idsnotcold

num_examples = len(idsnotcold)
indicesrand = np.arange(num_examples)
np.random.shuffle(indicesrand)
print "shuffle indices: ", indicesrand

print "batch atual = 1:4"
print

idxrord = indicesrand[1:4]
idxrord = sorted(idxrord)
print "shuffle indices batch ordenado:", idxrord

print
print "ids do batch com shuffle:"
print idsnotcold[idxrord]
print

notcoldfilt = np.array(idxnotcold)[idxrord]
print "specs batch shuffle: "
print np.array(specs)[notcoldfilt]

"""


