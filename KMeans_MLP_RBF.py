import numpy as np
import pandas as pd
import scipy
from sknn.mlp import Classifier, Regressor, Layer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.cluster import KMeans
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# Global Arrays
acc_RBF = []
prec_RBF = []
rec_RBF = []

# Global Learning Rate Array
alpha = [0.01]

for a in alpha:

    acc_Temp = []
    prec_Temp = []
    rec_Temp = []

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
        nn = Classifier(
            layers=[
                Layer("Sigmoid", units=100),
                Layer("Softmax", name="OutputLayer", units=cluster_size)
            ],
            learning_rate=a,
            n_iter=100
        )

        nn.fit(X_Train, pred)
        pred_RBF = nn.predict(X_Test)

        for m in range(0, len(pred_RBF)):
            if pred_RBF[m] == index:
                pred_RBF[m] = 1
            else:
                pred_RBF[m] = 0

        acc = accuracy_score(Y_Test, pred_RBF)
        prec = precision_score(Y_Test, pred_RBF)
        rec = recall_score(Y_Test, pred_RBF)

        acc_Temp.append(acc * 100)
        prec_Temp.append(prec)
        rec_Temp.append(rec)

    acc_RBF.append(sum(acc_Temp)/len(acc_Temp))
    prec_RBF.append(sum(prec_Temp)/len(prec_Temp))
    rec_RBF.append(sum(rec_Temp)/len(rec_Temp))

print acc_RBF
print prec_RBF
print rec_RBF

'''
#Graph Plotting

plotly.tools.set_credentials_file(username='dushyantRathore', api_key='5bzzn2vxpr')

# Create a trace
trace = go.Scatter(
    x = alpha,
    y = acc_RBF
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='line-MLP-RBF')'''




