import numpy as np
from sknn.mlp import Classifier, Layer

X = np.loadtxt('Created_Dataset.txt', usecols=[1, 2])
y = np.loadtxt('Created_Dataset.txt', usecols=[3])

# X_temp = np.array([[0,0],[1,1]])
# y_temp = np.array([1,1])

X_train_len = int(X.__len__() * 0.8)
X_test_len = X.__len__() - X_train_len

X_train_set = np.array(X[0:X_train_len])
X_test_set = np.array(X[X_train_len::])

y_train_set = np.array(y[0:X_train_len])
y_test_set = np.array(y[X_train_len::])

print("Lengths of X, X_train_set and X_test_set : ")

print(X.__len__())
print(X_train_set.__len__())
print(X_test_set.__len__())

# print(X_train_set)

#   Build the classifier
nn = Classifier(
    layers=[
        Layer("Rectifier", units=50),
        Layer("Softmax", name="OutputLayer", units=2)
    ],
    learning_rate=0.02,
    n_iter=100
)

# print(y)

#   Fit the classifier with training data set
nn.fit(X_train_set, y_train_set)

# parameters = nn.get_parameters()


#   Predict answers for test data set
y_pred = nn.predict(X_test_set)
#print(y_pred)
print(y_pred[2])
print(y_test_set[2])

if y_pred[0] == y_test_set[0]:
    print("Yay")


print("Now doing for the entire test set : ")
print("")

correctPredCount = 0
for i in range(0, X_test_set.__len__()):
    if y_pred[i] == y_test_set[i]:
        correctPredCount += 1

print("Reached here. lulz")

#   if and for loop ends

# print(parameters);
# print('The Final Prediction for [5, 10], [4.3, 20], [3.5, 25] is : ');
# print(y_pred);

print(correctPredCount)

accuracy = (float(correctPredCount) / X_test_set.__len__()) * 100

#print(accuracy)

print("Accuracy of prediction is : " + str(accuracy))
