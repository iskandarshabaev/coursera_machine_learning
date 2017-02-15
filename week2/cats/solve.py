import operator
import re

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import spatial


def make_word_list(stroke_list):
    word_list = []
    for stroke in stroke_list:
        words_in_sentence = list(filter(None, re.split('[^a-z]', stroke)))
        for word in words_in_sentence:
            word_list.append(word)
    return word_list


def make_dict(word_list):
    word_dict = {}
    k = 0
    for word in word_list:
        if word not in word_dict:
            word_dict[word] = k
            k += 1
    return word_dict


data = pd.read_csv('sentences.txt', header=None, sep='\n').values
data_lower = DataFrame([d[0].lower() for d in data]).values
stroke_list = [stroke[0] for stroke in data_lower]
word_list = make_word_list(stroke_list)
dict = make_dict(word_list)
m = np.zeros([22, 254])
for i in range(len(stroke_list)):
    stroke = stroke_list[i]
    words_in_sentence = list(filter(None, re.split('[^a-z]', stroke)))
    for word in dict:
        m[i][dict[word]] = words_in_sentence.count(word)

values = {}
for i in range(0, len(m)):
    values[i] = spatial.distance.cosine(m[21], m[i])
sorted_values = sorted(values.items(), key=operator.itemgetter(1))
for i in range(3):
    print(sorted_values[i])
