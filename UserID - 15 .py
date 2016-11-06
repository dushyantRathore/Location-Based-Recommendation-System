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

f = pd.read_csv("UserID 15 Trimmed_Dataset (Random - 40).csv", delimiter = ",")
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

Y_Tr = Train["Y_Value"]
Y_list1 = list(Y_Tr)
Y_Train = np.asarray(Y_list1)
#print Y_list
#print len(Y_Train)

X_Test_DF = Test.ix[:, 2:5]
X_Test = X_Test_DF.as_matrix()
#print X_Test
#print len(X_Test)

Y_Te = Test["Y_Value"]
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

print "Count Train : " + str(count_Train)
print "Count Test : " + str(count_Test)



# Support Vector Machine - rbf
clf_svm_rbf = svm.SVC(kernel='rbf')
clf_svm_rbf.fit(X_Train, Y_Train)
y_pred_svm_rbf = clf_svm_rbf.predict(X_Test)
score_svm_rbf = accuracy_score(Y_Test, y_pred_svm_rbf)

count_svm_rbf = 0
for i in y_pred_svm_rbf:
    if i == 1:
        count_svm_rbf = count_svm_rbf + 1

print "\nThe no. of 1 in prediction set of SVM rbf Kernel : " + str(count_svm_rbf)
print "The Accuracy for SVM rbf Kernel is : " + str(score_svm_rbf * 100)



# Probabilistic Neural Network - Neupy
nw = algorithms.PNN(std=0.1, verbose=False)
nw.train(X_Train, Y_Train)
y_pred_PNN = nw.predict(X_Test)
score_PNN = accuracy_score(Y_Test, y_pred_PNN)

count_pnn = 0
for i in y_pred_PNN:
    if i == 1:
        count_pnn = count_pnn + 1

print "\nThe no. of 1 in prediction set of PNN : " + str(count_pnn)
print "The Accuracy for PNN is : " + str(score_PNN * 100)


