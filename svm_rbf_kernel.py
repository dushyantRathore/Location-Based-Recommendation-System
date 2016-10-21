from sklearn import svm
import numpy as np

X = np.loadtxt('Created_Dataset.txt',usecols=[1,2])
y = np.loadtxt('Created_Dataset.txt',usecols=[3])



X_training_length = int(0.8*X.__len__())
y_training_length = int(0.8*y.__len__())

print X_training_length 
print y_training_length

X_test_length = X.__len__() - X_training_length
y_test_length = y.__len__() - y_training_length


X_training_set = np.array(X[0:y_training_length])
y_training_set = np.array(y[0:y_training_length])

X_testing_set = np.array(X[X_training_length:])
y_testing_set = np.array(y[y_training_length:])

# clf is svm object
clf = svm.SVC()
clf.fit(X_training_set, y_training_set)

y_prediction_set = clf.predict(X_testing_set)

pred_correct_count = 0

for i in range(0,y_test_length):
	if y_prediction_set[i]==y_testing_set[i]:
		pred_correct_count+=1

accuracy = (float(pred_correct_count)/float(y_test_length))*100

print "Accuracy with SVM using RBF kernel is : "+ str(accuracy)



