# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 20:18:43 2017

@author: WHY
"""

import numpy as np
from sklearn import preprocessing , cross_validation , svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
print(df.head(5))
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, 
                                                            test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
print(clf.n_support_)