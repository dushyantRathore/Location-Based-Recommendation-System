import numpy as np
import pandas as pd
import scipy
from sknn.mlp import Classifier, Regressor, Layer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn import svm
from neupy import algorithms, environment

f = pd.read_csv("UserID 15 Trimmed_Dataset (Random).csv", delimiter = ",")
df = pd.DataFrame(f)
#print df

# Form Training and Testing Dataset
Train, Test = train_test_split(df, test_size=0.2)

# Print the length of the training and Testing Dataset
#print len(Train)
#print len(Test)

X_Train_DF = Train.ix[:, 2:5]
X_Train = X_Train_DF.as_matrix()
#print X_Train
#print len(X_Train)

Y_Tr = Train["Y.value"]
Y_list1 = list(Y_Tr)
Y_Train = np.asarray(Y_list1)
#print Y_list
#print len(Y_Train)

X_Test_DF = Test.ix[:, 2:5]
X_Test = X_Test_DF.as_matrix()
#print X_Test
#print len(X_Test)

Y_Te = Test["Y.value"]
Y_list2 = list(Y_Te)
Y_Test = np.asarray(Y_list2)
#print Y_list
#print len(Y_Train)

count_Train = 0
count_Test = 0

for i in Y_Train:
    if i == 1:
        count_Train = count_Train + 1

for i in Y_Test:
    if i == 1:
        count_Test = count_Test + 1

print count_Train
print count_Test

# Support Vector Machine
clf_svm = svm.SVC()
clf_svm.fit(X_Train, Y_Train)
y_pred_svm = clf_svm.predict(X_Test)
score_svm = accuracy_score(Y_Test, y_pred_svm)

count_svm = 0
for i in y_pred_svm:
    if i == 1:
        count_svm = count_svm + 1

print "\nThe no. of 1 in prediction set of SVM : "
print count_svm
print "The Accuracy for SVM Kernel is  "
print score_svm * 100

# Probabilistic Neural Network - Neupy
nw = algorithms.PNN(std=0.1, verbose=False)
nw.train(X_Train, Y_Train)
y_pred_PNN = nw.predict(X_Test)
score_PNN = accuracy_score(Y_Test, y_pred_PNN)

count_pnn = 0
for i in y_pred_PNN:
    if i == 1:
        count_pnn = count_pnn + 1

print "\nThe no. of 1 in prediction set of PNN : "
print count_pnn
print "The Accuracy for PNN is : "
print score_PNN * 100










