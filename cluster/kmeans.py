# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 22:47:05 2017

@author: WHY
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')

X = np.array([[1,6],
             [3,2],
             [5,9],
             [10,21],
             [11,26],
             [12,16]])

plt.scatter(X[:,0],X[:,1], s= 150)
plt.show()

clf = KMeans(n_clusters = 2)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_
print('centroids: ',centroids)
print('labels: ',labels)

colors = ["g.","r.","c.","b.","k."]
for i in range(len(X)):
    plt.plot(X[i][0],X[i][1], colors[labels[i]], markersize = 20)
plt.scatter(centroids[:,0],centroids[:,1],marker ='x', s=150)
plt.show()