from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn import svm

acc_SVM = []
prec_SVM = []
rec_SVM = []

for i in range (1,9):

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

    kmeans = KMeans(n_clusters = cluster_size, random_state=0).fit(X_Train)
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

    # print cluster_array

    # SVM - RBF Kernel
    svm_RBF = svm.SVC()
    svm_RBF.fit(X_Train, pred)
    svm_pred = svm_RBF.predict(X_Test)

    for m in range(0, len(svm_pred)):
        if svm_pred[m] == index:
            svm_pred[m] = 1
        else:
            svm_pred[m] = 0

    acc = accuracy_score(Y_Test, svm_pred)
    rec = recall_score(Y_Test, svm_pred)
    prec = precision_score(Y_Test, svm_pred)

    acc_SVM.append(acc * 100)
    rec_SVM.append(rec)
    prec_SVM.append(prec)


print "Accuracy : " + str(sum(acc_SVM)/len(acc_SVM))
print "Precision : " + str(sum(prec_SVM)/len(prec_SVM))
print "Recall : " + str(sum(rec_SVM)/len(rec_SVM))





