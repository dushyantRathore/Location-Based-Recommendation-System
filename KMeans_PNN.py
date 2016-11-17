from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn import svm
from neupy import algorithms, environment
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# Global Arrays
mclass_PNN = []
prec_PNN = []
rec_PNN = []


sigma = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

for s in sigma:

    mclass_PNN_temp = []
    prec_PNN_temp = []
    rec_PNN_temp = []

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

        # print "The no of 1's in Training Set : " + str(c)

        # Form the Testing Set
        X_Test_DF = Test.ix[:, 2:5]
        X_Test = X_Test_DF.as_matrix()
        Y_Te = Test["Y_Value"]
        Y_list2 = list(Y_Te)
        Y_Test = np.asarray(Y_list2)

        kmeans = KMeans(n_clusters = cluster_size, random_state=0).fit(X_Train)
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

        # print cluster_array

        # PNN
        clf_PNN = algorithms.PNN(std=s)
        clf_PNN.fit(X_Train, pred)
        y_pred = clf_PNN.predict(X_Test)

        for m in range(0, len(y_pred)):
            if y_pred[m] == index:
                y_pred[m] = 1
            else:
                y_pred[m] = 0

        acc = accuracy_score(Y_Test, y_pred)
        prec = precision_score(Y_Test, y_pred)
        rec = recall_score(Y_Test, y_pred)

        mclass_PNN_temp.append(100 - (acc * 100))
        prec_PNN_temp.append(prec)
        rec_PNN_temp.append(rec)

    mclass_PNN.append(sum(mclass_PNN_temp)/len(mclass_PNN_temp))
    prec_PNN.append(sum(prec_PNN_temp)/len(prec_PNN_temp))
    rec_PNN.append(sum(rec_PNN_temp)/len(rec_PNN_temp))

print "Misclassification : "
print mclass_PNN
print "Precision : "
print prec_PNN
print "Recall : "
print rec_PNN


# Graph Plotting

plotly.tools.set_credentials_file(username='dushyantRathore', api_key='5bzzn2vxpr')

# Create a trace
trace = go.Scatter(
    x=sigma,
    y=mclass_PNN
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='PNN-Sigma_Values')
