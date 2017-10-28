# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 19:13:25 2017

@author: WHY
"""
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
import pandas as pd
import random
from pdb import set_trace as bp

style.use('fivethirtyeight')

#dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
#new_feature = [5,7]



def knn(data, predict, k=3):
        if len(data)>=k:
            warnings.warn('u idiot')
        distances = []
        for group in data:
            for features in data[group]:
                ed = np.linalg.norm(np.array(features)-np.array(predict))
                distances.append([ed,group])
#        print('distances = ',distances)
        votes = [i[1] for i in sorted(distances)[:3]]
#        print('votes = ',votes)
        vote_result = Counter(votes).most_common(1)[0][0]
#        print('vote_result = ',vote_result)
        confidence = Counter(votes).most_common(1)[0][1]/k
        return vote_result, confidence

#result = knn(dataset, new_feature, k=3)
#print('result = ',result)
#[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_feature[0],new_feature[1],s=100, color='g')
#plt.show()

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
print(full_data[:5])
random.shuffle(full_data)
print(20*'$')
print(full_data[:5])

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
    
for i in test_data:
    print('i ', i)
    print('test_set b', test_set)
    test_set[i[-1]].append(i[:-1])
    print('test_set a', test_set)


correct = 0
total = 0

for group in test_set:
        for data in test_set[group]:
            vote, confidence = knn(train_set, data, k=5)
            if group == vote:
                correct+=1
            total+=1
print('Accuracy: ',correct/total)
print('confidence: ',confidence)