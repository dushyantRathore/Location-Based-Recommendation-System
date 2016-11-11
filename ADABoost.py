import numpy as np
import pandas as pd
import scipy
from sknn.mlp import Classifier, Regressor, Layer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier

#Global Arrays
acc_ADA = []
prec_ADA = []
rec_ADA = []

for i in range(1, 9):

    x = "/home/dushyant/Desktop/Minor/Codes/Foursquare Section/Cracked/Dataset/user" + str(i) + ".csv"
    f = pd.read_csv(x, delimiter=",")
    df = pd.DataFrame(f)

    Train, Test = train_test_split(df, test_size=0.3)

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

    #print "The no of 1's in Training Set : " + str(c)

    # Form the Testing Set
    X_Test_DF = Test.ix[:, 2:5]
    X_Test = X_Test_DF.as_matrix()
    Y_Te = Test["Y_Value"]
    Y_list2 = list(Y_Te)
    Y_Test = np.asarray(Y_list2)

    kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(X_Train)
    #print len(kmeans.labels_)

    pred = kmeans.predict(X_Train)

    #print "Prediction Array : "
    #print pred

    for k in range(0, len(pred)):
        if Y_Train[k] == 1:
            cluster_array[pred[k]] += 1

    #print "Cluster Array : "
    #print cluster_array

    m = max(cluster_array)

    index = 0
    for l in range(0, cluster_size):
        if cluster_array[l] == m:
            index = l

    #print "Index Value : " + str(index)

    # MLP
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(X_Train, pred)
    pred_ada = clf.predict(X_Test)

    for m in range(0, len(pred_ada)):
        if pred_ada[m] == index:
            pred_ada[m] = 1
        else:
            pred_ada[m] = 0

    acc = accuracy_score(Y_Test, pred_ada)
    prec = precision_score(Y_Test, pred_ada)
    rec = recall_score(Y_Test, pred_ada)

    acc_ADA.append(acc * 100)
    prec_ADA.append(prec)
    rec_ADA.append(rec)


print "Accuracy : " + str(sum(acc_ADA)/len(acc_ADA))
print "Precision : " + str(sum(prec_ADA)/len(prec_ADA))
print "Recall : " + str(sum(rec_ADA)/len(rec_ADA))