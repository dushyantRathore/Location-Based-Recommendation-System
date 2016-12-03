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

# Global Arrays
mclass_ADA = []
prec_ADA = []
rec_ADA = []

learning_rate = [0.05, 0.1, 0.3, 0.5, 0.7, 1, 2, 5]

for alpha in learning_rate:

    mclass_ADA_temp = []
    prec_ADA_temp =[]
    rec_ADA_temp =[]

    for i in range(1, 15):

        x = "/home/dushyant/Desktop/Minor/Code/Updated_Dataset/user" + str(i) + ".csv"
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

        # Gradient Classifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=alpha)
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

        mclass_ADA_temp.append(100 - (acc * 100))
        prec_ADA_temp.append(prec)
        rec_ADA_temp.append(rec)

    mclass_ADA.append(sum(mclass_ADA_temp)/len(mclass_ADA_temp))
    prec_ADA.append(sum(prec_ADA_temp)/len(prec_ADA_temp))
    rec_ADA.append(sum(rec_ADA_temp)/len(rec_ADA_temp))

print "Misclassification : "
print mclass_ADA
print "Precision : "
print prec_ADA
print "Recall : "
print rec_ADA

'''
# Graph Plotting
plotly.tools.set_credentials_file(username='dushyantRathore', api_key='5bzzn2vxpr')

# Create a trace
trace = go.Scatter(
    x=learning_rate,
    y=mclass_ADA
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='ADABoost - Learning_Rate')
'''