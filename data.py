import os
from eunjeon import Mecab

def pos(text):
    mecab = Mecab()
    res = mecab.pos(text)
    return res

def extractNouns(pos, vocaWeight):
    start = 0
    len_res = len(pos)
    nouns = []
    tokenized = []
    while start < len_res:
        if start + 2 < len_res:
            three_part = pos[start][0] + pos[start + 1][0] + pos[start + 2][0]
            if vocaWeight.get(three_part, None) != None:
                tokenized.append(three_part)
                nouns.append(three_part)
                start += 3
                continue
        if start + 1 < len_res:
            two_part = pos[start][0] + pos[start + 1][0]
            if vocaWeight.get(two_part, None) != None:
                tokenized.append(two_part)
                nouns.append(two_part)
                start += 2
                continue
        first = pos[start][0]
        t = pos[start][1]
        tokenized.append(first)
        if t == 'NNG':
            if len(first) > 1:
                nouns.append(first)
        start += 1

    return nouns

def getInput():
    vocaWeight = getVocaWeight()
    data = []
    with open(os.path.join('./data/input.txt'), 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            d = extractNouns(pos(line), vocaWeight)
            t = " ".join([dd for dd in d])
            data.append(t)

    return data[0]

def getKerasTestData():
    vocaWeight = getVocaWeight()
    data = []
    with open(os.path.join('./data/input.txt'), 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            d = extractNouns(pos(line), vocaWeight)
            data.append(d)

    return data

def getData():
    vocaWeight = getVocaWeight()
    data = []
    with open(os.path.join('./data/vp_text.txt'), 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            d = extractNouns(pos(line), vocaWeight)
            t = " ".join([dd for dd in d])
            data.append(t)

    return data, vocaWeight

def getKerasVpData():
    vocaWeight = getVocaWeight()
    data = []
    with open(os.path.join('./data/vp_text.txt'), 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            d = extractNouns(pos(line), vocaWeight)
            data.append(d)

    return data, vocaWeight

def getKerasNormalData():
    vocaWeight = getVocaWeight()
    data = []
    with open(os.path.join('./data/normal_text.txt'), 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            d = extractNouns(pos(line), vocaWeight)
            data.append(d)

    return data, vocaWeight

def getKerasTrainingData():
    x_data = []
    y_data = []
    with open(os.path.join('./keras_lstm/data/training_data.txt'), 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            x_data.append(eval(line.replace('\n', '')))

    with open(os.path.join('./keras_lstm/data/label.txt'), 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            y_data.append(int(line.replace('\n', '')))

    return x_data, y_data

def getVocaWeight():
    vocaWeight = {}
    with open(os.path.join('./data/voca_data.txt'), 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            v = line.split("^")
            vocaWeight[v[0]] = v[3]

    return vocaWeight