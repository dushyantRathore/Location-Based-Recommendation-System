import numpy as np
import pandas as pd
import scipy
from sknn.mlp import Classifier, Regressor, Layer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.cluster import KMeans

f = pd.read_csv("/home/dushyant/Desktop/Minor/Codes/Foursquare Section/Cracked/UserID 15 - Original.csv", delimiter=",")
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

print "The no of 1's in Training Set : " + str(c)

# Form the Testing Set
X_Test_DF = Test.ix[:, 2:5]
X_Test = X_Test_DF.as_matrix()
Y_Te = Test["Y_Value"]
Y_list2 = list(Y_Te)
Y_Test = np.asarray(Y_list2)

kmeans = KMeans(n_clusters = cluster_size, random_state=0).fit(X_Train)
print len(kmeans.labels_)

pred = kmeans.predict(X_Train)

print "Prediction Array : "
print pred

for i in range(0, len(pred)):
    if Y_Train[i] == 1:
        cluster_array[pred[i]] += 1

print "Cluster Array : "
print cluster_array

m = max(cluster_array)

index = 0
for i in range(0, cluster_size):
    if cluster_array[i] == m:
        index = i

print "Index Value : " + str(index)

# print cluster_array

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

for i in range(0, len(pred_RBF)):
    if pred_RBF[i] == index:
        pred_RBF[i] = 1
    else:
        pred_RBF[i] = 0

acc = accuracy_score(Y_Test, pred_RBF)
rec = recall_score(Y_Test, pred_RBF)
prec = precision_score(Y_Test, pred_RBF)

print "Accuracy : " + str(acc)
print "Precision : " + str(prec)
print "Recall : " + str(rec)





