import numpy as np
import pandas as pd
import scipy
from sknn.mlp import Classifier, Regressor, Layer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from neupy import algorithms, environment

# Global Arrays
mclass_PNN = []
prec_PNN = []
rec_PNN = []

mclass_ADA = []
prec_ADA = []
rec_ADA = []

clusters = [5, 10, 15, 20]

for cluster_size in clusters:

    mclass_PNN_temp = []
    prec_PNN_temp = []
    rec_PNN_temp = []

    mclass_ADA_temp = []
    prec_ADA_temp = []
    rec_ADA_temp = []

    for i in range(1, 15):

        x = "/home/dushyant/Desktop/Minor/Code/Updated_Dataset/user" + str(i) + ".csv"
        f = pd.read_csv(x, delimiter=",")
        df = pd.DataFrame(f)

        Train, Test = train_test_split(df, test_size=0.4)

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
        # print cluster_array

        # PNN
        clf_PNN = algorithms.PNN(std=0.1)
        clf_PNN.fit(X_Train, pred)
        y_pred_PNN = clf_PNN.predict(X_Test)

        for m in range(0, len(y_pred_PNN)):
            if y_pred_PNN[m] == index:
                y_pred_PNN[m] = 1
            else:
                y_pred_PNN[m] = 0

        acc_p = accuracy_score(Y_Test, y_pred_PNN)
        prec_p = precision_score(Y_Test, y_pred_PNN)
        rec_p = recall_score(Y_Test, y_pred_PNN)

        mclass_PNN_temp.append(100 - (acc_p * 100))
        prec_PNN_temp.append(prec_p)
        rec_PNN_temp.append(rec_p)


        # Gradient Classifier
        clf_ADA = GradientBoostingClassifier(n_estimators=100)
        clf_ADA.fit(X_Train, pred)
        y_pred_ADA = clf_ADA.predict(X_Test)

        for m in range(0, len(y_pred_ADA)):
            if y_pred_ADA[m] == index:
                y_pred_ADA[m] = 1
            else:
                y_pred_ADA[m] = 0

        acc_a = accuracy_score(Y_Test, y_pred_ADA)
        prec_a = precision_score(Y_Test, y_pred_ADA)
        rec_a = recall_score(Y_Test, y_pred_ADA)

        mclass_ADA_temp.append(100 - (acc_a * 100))
        prec_ADA_temp.append(prec_a)
        rec_ADA_temp.append(rec_a)


    mclass_PNN.append(sum(mclass_PNN_temp)/len(mclass_PNN_temp))
    prec_PNN.append(sum(prec_PNN_temp)/len(prec_PNN_temp))
    rec_PNN.append(sum(rec_PNN_temp)/len(rec_PNN_temp))

    mclass_ADA.append(sum(mclass_ADA_temp)/len(mclass_ADA_temp))
    prec_ADA.append(sum(prec_ADA_temp)/len(prec_ADA_temp))
    rec_ADA.append(sum(rec_ADA_temp)/len(rec_ADA_temp))

'''
#Graph Plotting
plotly.tools.set_credentials_file(username='dushyantRathore', api_key='5bzzn2vxpr')

# Create traces

trace1 = go.Scatter(
    x=clusters,
    y=mclass_PNN,
    mode='lines',
    name='PNN'
)
trace2 = go.Scatter(
    x=clusters,
    y=mclass_ADA,
    mode='lines',
    name='ADABoost'
)
data = [trace1, trace2]

# Plot and embed in ipython notebook!
py.iplot(data, filename='PNN vs ADA - Cluster Size')
'''
