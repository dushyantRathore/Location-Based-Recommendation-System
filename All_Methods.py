import numpy as np
import pandas as pd
import scipy
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.cluster import KMeans
from sknn.mlp import Classifier, Regressor, Layer
from sklearn import svm
from neupy import algorithms, environment
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# Global Arrays
mclass_RBF = []
mclass_SVM = []
mclass_PNN = []
mclass_ADA = []

rec_RBF = []
rec_SVM = []
rec_PNN = []
rec_ADA = []

prec_RBF = []
prec_SVM = []
prec_PNN = []
prec_ADA = []


for i in range(1, 15):
    x = "/home/dushyant/Desktop/Minor/Code/Updated_Dataset/user" + str(i) + ".csv"
    f = pd.read_csv(x, delimiter=",")
    df = pd.DataFrame(f)

    Train, Test = train_test_split(df, test_size=0.3)

    # Fixed Cluster Size
    cluster_size = 35
    cluster_array = np.zeros(cluster_size)

    # Form the Training Set
    X_Train_DF = Train.ix[:, 2:5]
    X_Train = X_Train_DF.as_matrix()
    Y_Tr = Train["Y_Value"]
    Y_list1 = list(Y_Tr)
    Y_Train = np.asarray(Y_list1)

    c = 0
    for j in Y_Train:
        if j == 1:
            c = c + 1

    # print "The no of 1's in Training Set : " + str(c)

    # Form the Testing Set
    X_Test_DF = Test.ix[:, 2:5]
    X_Test = X_Test_DF.as_matrix()
    Y_Te = Test["Y_Value"]
    Y_list2 = list(Y_Te)
    Y_Test = np.asarray(Y_list2)

    kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(X_Train)
    # print len(kmeans.labels_)

    pred = kmeans.predict(X_Train)

    # print "Prediction Array : "
    # print pred

    for k in range(0, len(pred)):
        if Y_Train[k] == 1:
            cluster_array[pred[k]] += 1

    # print "Cluster Array : "
    # print cluster_array

    m = max(cluster_array)

    index = 0
    for l in range(0, cluster_size):
        if cluster_array[l] == m:
            index = l

    # print "Index Value : " + str(index)


    # MLP
    nn = Classifier(
        layers=[
            Layer("Sigmoid", units=100),
            Layer("Softmax", name="OutputLayer", units=cluster_size)
        ],
        learning_rate=0.01,
        n_iter=100
    )

    nn.fit(X_Train, pred)
    pred_RBF = nn.predict(X_Test)

    for m in range(0, len(pred_RBF)):
        if pred_RBF[m] == index:
            pred_RBF[m] = 1
        else:
            pred_RBF[m] = 0

    acc_r = accuracy_score(Y_Test, pred_RBF)
    rec_r = recall_score(Y_Test,pred_RBF)
    prec_r = precision_score(Y_Test, pred_RBF)
    mclass_RBF.append(100 - (acc_r * 100))
    rec_RBF.append(rec_r)
    prec_RBF.append(prec_r)


    # SVM - RBF Kernel
    svm_RBF = svm.SVC()
    svm_RBF.fit(X_Train, pred)
    svm_pred = svm_RBF.predict(X_Test)

    for m in range(0, len(svm_pred)):
        if svm_pred[m] == index:
            svm_pred[m] = 1
        else:
            svm_pred[m] = 0

    acc_s = accuracy_score(Y_Test, svm_pred)
    rec_s = recall_score(Y_Test, svm_pred)
    prec_s = precision_score(Y_Test, svm_pred)
    mclass_SVM.append(100 - (acc_s * 100))
    rec_SVM.append(rec_s)
    prec_SVM.append(prec_s)


    # PNN
    clf_PNN = algorithms.PNN(std=0.1)
    clf_PNN.fit(X_Train, pred)
    pred_PNN = clf_PNN.predict(X_Test)

    for m in range(0, len(pred_PNN)):
        if pred_PNN [m] == index:
            pred_PNN[m] = 1
        else:
            pred_PNN[m] = 0

    acc_p = accuracy_score(Y_Test, pred_PNN)
    rec_p = recall_score(Y_Test, pred_PNN)
    prec_p = precision_score(Y_Test, pred_PNN)
    mclass_PNN.append(100 - (acc_p * 100))
    rec_PNN.append(rec_p)
    prec_PNN.append(prec_p)


    # Gradient Classifier
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(X_Train, pred)
    pred_ada = clf.predict(X_Test)

    for m in range(0, len(pred_ada)):
        if pred_ada[m] == index:
            pred_ada[m] = 1
        else:
            pred_ada[m] = 0

    acc_g = accuracy_score(Y_Test, pred_ada)
    rec_g = recall_score(Y_Test, pred_ada)
    prec_g = precision_score(Y_Test, pred_ada)
    mclass_ADA.append(100 - (acc_g * 100))
    rec_ADA.append(rec_g)
    prec_ADA.append(prec_g)

print "RBF"
print sum(mclass_RBF)/len(mclass_RBF)
print "SVM"
print sum(mclass_SVM)/len(mclass_SVM)
print "PNN"
print sum(mclass_PNN)/len(mclass_PNN)
print "ADABoost"
print sum(mclass_ADA)/len(mclass_ADA)


'''
# Graph Plotting
plotly.tools.set_credentials_file(username='dushyantRathore', api_key='5bzzn2vxpr')

x_axis = ['RBF NN', 'SVM', 'PNN', 'AdaBoostClassifier']

# Misclassification %
y_axis_mclass = []
y_axis_mclass.append(sum(mclass_RBF)/len(mclass_RBF))
y_axis_mclass.append(sum(mclass_SVM)/len(mclass_SVM))
y_axis_mclass.append(sum(mclass_PNN)/len(mclass_PNN))
y_axis_mclass.append(sum(mclass_ADA)/len(mclass_ADA))

data1 = [go.Bar(
            x=x_axis,
            y=y_axis_mclass
    )]

py.iplot(data1, filename='All Methods - Misclassification %')

# Recall
y_axis_recall = []
y_axis_recall.append(sum(rec_RBF)/len(rec_RBF))
y_axis_recall.append(sum(rec_SVM)/len(rec_SVM))
y_axis_recall.append(sum(rec_PNN)/len(rec_PNN))
y_axis_recall.append(sum(rec_ADA)/len(rec_ADA))

data2 = [go.Bar(
            x=x_axis,
            y=y_axis_recall
    )]

py.iplot(data2, filename='All Methods - Recall')

# Precision
y_axis_precision = []
y_axis_precision.append(sum(prec_RBF)/len(prec_RBF))
y_axis_precision.append(sum(prec_SVM)/len(prec_SVM))
y_axis_precision.append(sum(prec_PNN)/len(prec_PNN))
y_axis_precision.append(sum(prec_ADA)/len(prec_ADA))

data3 = [go.Bar(
            x=x_axis,
            y=y_axis_precision
    )]

py.iplot(data3, filename='All Methods - Precision')
'''