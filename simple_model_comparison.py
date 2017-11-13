# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:17:41 2017

@author: Heikki Niittylä
"""

from sklearn import neural_network 
from sklearn import svm
from sklearn import neighbors
from sklearn import gaussian_process
from sklearn import tree
from sklearn import ensemble

import matplotlib.pyplot as plt
import numpy as np

#Learning data
learn_min = -5.0
learn_max = 5.0
learn_step = 0.5

X = [[i] for i in np.arange(learn_min, learn_max, learn_step)]

# A few example functions to get some nice curves
#y = [i for i in X]
y = [np.sign(i)[0] for i in X]
#y = [np.exp(i)[0] for i in X]
#y = [np.sin(i)[0] for i in X]


#Train different kind of models

MLP = neural_network.MLPRegressor()
NSV = svm.NuSVR()
KN = neighbors.KNeighborsRegressor()
GPR = gaussian_process.GaussianProcessRegressor()
DT = tree.DecisionTreeRegressor()
GB = ensemble.GradientBoostingRegressor()

MLP.fit(X, y)
NSV.fit(X, y)
KN.fit(X, y)
GPR.fit(X, y)
DT.fit(X, y)
GB.fit(X, y)

#Plot
test_points = [[i] for i in np.arange(-5.0, 5.0, 0.05)]

plt.figure(figsize=(12,8))
plt.plot(X, y, label="Original")

plt.plot(test_points, MLP.predict(test_points), label="MLP")
plt.plot(test_points, NSV.predict(test_points), label="NSV")
plt.plot(test_points, KN.predict(test_points), label="KN")
plt.plot(test_points, GPR.predict(test_points), label="GPR")
plt.plot(test_points, DT.predict(test_points), label="DT")
plt.plot(test_points, GB.predict(test_points), label="GB")

plt.legend()
plt.grid()
plt.show()

# Calculate and print how much each model actually differs from original data
print("Total errors:")
print("MLP: {:06.9f}".format(np.absolute(MLP.predict(X)-y).sum()))
print("NSV: {:06.9f}".format(np.absolute(NSV.predict(X)-y).sum()))
print("KN : {:06.9f}".format(np.absolute(KN.predict(X)-y).sum()))
print("GPR: {:06.9f}".format(np.absolute(GPR.predict(X)-y).sum()))
print("DT : {:06.9f}".format(np.absolute(DT.predict(X)-y).sum()))
print("GB : {:06.9f}".format(np.absolute(GB.predict(X)-y).sum()))
