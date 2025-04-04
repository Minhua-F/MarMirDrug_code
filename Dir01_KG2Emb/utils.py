import os
from copy import deepcopy
import pickle
import random
import numpy as np
import time
import datetime
from itertools import groupby

import loss

class Triple(object):
    def __init__(self, head, tail, relation):
        self.h = head
        self.t = tail
        self.r = relation

# Compare two Triples in the order of head, relation and tail
def cmp_head(a, b):
    return (a.h < b.h or (a.h == b.h and a.r < b.r) or (a.h == b.h and a.r == b.r and a.t < b.t))

# Compare two Triples in the order of tail, relation and head
def cmp_tail(a, b):
    return (a.t < b.t or (a.t == b.t and a.r < b.r) or (a.t == b.t and a.r == b.r and a.h < b.h))

# Compare two Triples in the order of relation, head and tail
def cmp_rel(a, b):
    return (a.r < b.r or (a.r == b.r and a.h < b.h) or (a.r == b.r and a.h == b.h and a.t < b.t))

def minimal(a, b):
    if a > b:
        return b
    return a

def cmp_list(a, b):
    return (minimal(a.h, a.t) > minimal(b.h, b.t))

# Write a list of Triples into a file, with three numbers (head tail relation) per line
def process_list(tripleList, dataset, filename):
    with open(os.path.join('./datasets/', dataset, filename).replace('\\', '/'), 'w') as fw:
        fw.write(str(len(tripleList)) + '\n')
        for triple in tripleList:
            fw.write(str(triple.h) + '\t' + str(triple.t) + '\t' + str(triple.r) + '\n')

emptyTriple = Triple(0, 0, 0)

def getRel(triple):
    return triple.r

# Gets the number of entities/relations/triples
def getAnythingTotal(inPath, fileName):  # read the total number in line 1
    with open(os.path.join(inPath, fileName).replace('\\', '/'), 'r') as fr:
        for line in fr:
            return int(line)

def loadTriple(inPath, fileName):
    with open(os.path.join(inPath, fileName).replace('\\', '/'), 'r') as fr:
        i = 0
        tripleList = []
        for line in fr:
            if i == 0:
                tripleTotal = int(line)
                i += 1
            else:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[1])
                rel = int(line_split[2])
                tripleList.append(Triple(head, tail, rel))

    tripleDict = {}
    for triple in tripleList:
        tripleDict[(triple.h, triple.t, triple.r)] = True

    return tripleTotal, tripleList, tripleDict

# Calculate the statistics of datasets
def calculate_one_or_many(dataset):
    tripleTotal, tripleList, tripleDict = loadTriple(dataset, 'triple2id.txt')
    # You should sort first before groupby!
    tripleList.sort(key=lambda x: (x.r, x.h, x.t))
    grouped = [(k, list(g)) for k, g in groupby(tripleList, key=getRel)]
    num_of_relations = len(grouped)
    head_per_tail_list = [0] * num_of_relations
    tail_per_head_list = [0] * num_of_relations

    one_to_one = []
    one_to_many = []
    many_to_one = []
    many_to_many = []

    for elem in grouped:
        headList = []
        tailList = []
        for triple in elem[1]:
            headList.append(triple.h)
            tailList.append(triple.t)
        headSet = set(headList)
        tailSet = set(tailList)
        head_per_tail = len(headList) / len(tailSet)
        tail_per_head = len(tailList) / len(headSet)
        if elem[0] >= len(head_per_tail_list):
            continue
        head_per_tail_list[elem[0]] = head_per_tail
        tail_per_head_list[elem[0]] = tail_per_head
        if head_per_tail < 1.5 and tail_per_head < 1.5:
            one_to_one.append(elem[0])
        elif head_per_tail >= 1.5 and tail_per_head < 1.5:
            many_to_one.append(elem[0])
        elif head_per_tail < 1.5 and tail_per_head >= 1.5:
            one_to_many.append(elem[0])
        else:
            many_to_many.append(elem[0])

    with open(os.path.join(dataset, 'head_tail_proportion.pkl').replace('\\', '/'), 'wb') as fw:
        pickle.dump(tail_per_head_list, fw)
        pickle.dump(head_per_tail_list, fw)

def which_loss_type(num):
    if num == 0:
        return loss.marginLoss
    elif num == 1:
        return loss.EMLoss
    elif num == 2:
        return loss.WGANLoss
    elif num == 3:
        return nn.MSELoss