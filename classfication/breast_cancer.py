# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:38:50 2017

@author: WHY
"""
import numpy as np
from sklearn import preprocessing , cross_validation , neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
print(df.head(5))
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
print('after drop id: \n',df.head(5))


X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, 
                                                            test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_mesurement = np.array([[4,2,1,1,1,2,3,2,1],[2,2,1,1,1,2,3,2,1]])
example_mesurement = example_mesurement.reshape(len(example_mesurement), -1)

prediction = clf.predict(example_mesurement)
print(prediction)