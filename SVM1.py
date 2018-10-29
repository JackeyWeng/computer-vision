# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:33:33 2018

@author: 付成
"""



########## data loading and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

f=open('D:\python\watermelon_4.2.txt')
df=f.readlines()
dfLables = ['id', 'density', 'sugar_content', 'label']


# plt.figure(0)
# sns.FacetGrid(df, hue='label', size=5).map(plt.scatter, 'density', 'sugar_content').add_legend() 
# plt.show()

X = df[['density', 'sugar_content']].values
y = df['label'].values

########## SVM training and comparison
# based on linear kernel as well as gaussian kernel
from sklearn import svm 

for fig_num, kernel in enumerate(('linear', 'rbf')): 
    # initial
    svc = svm.SVC(C=1000, kernel=kernel)  # classifier 1 based on linear kernel
    # train
    svc.fit(X, y)
    # get support vectors
    sv = svc.support_vectors_
    
    ##### draw decision zone
    plt.figure(fig_num)
    plt.clf()
    
    # plot point and mark out support vectors
    plt.scatter( X[:,0],  X[:,1], edgecolors='k', c=y, cmap=plt.cm.Paired,  zorder=10)
    plt.scatter(sv[:,0], sv[:,1], edgecolors='k', facecolors='none', s=80, linewidths=2, zorder=10)
    
    # plot the decision boundary and decision zone into a color plot
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    XX, YY = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svc.decision_function(np.c_[XX.ravel(), YY.ravel()]) 
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z>0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    
    plt.title(kernel)
    plt.axis('tight')
    
plt.show()

print(' - PY131 - ')