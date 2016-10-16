import numpy as np
import pandas as pd
import scipy
from sknn.mlp import Classifier, Regressor, Layer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# Dataframe Formation
f = pd.read_csv("Created_Dataset.csv", delimiter = ",")
df = pd.DataFrame(f)

# Rating/Count Array Formation
x = df.ix[:, 1:3]
X = np.array(x)

# Y-Value array formation
y = df.ix[:, 3:]
y_list = list(df["Y-Value"])
Y = np.asarray(y_list)

# Calculate the Train and Test Lengths
X_train_len = int(X.__len__() * 0.8)
X_test_len = X.__len__() - X_train_len

# Training/Testing Set Formation
X_train_set = X[0:X_train_len]
X_test_set = X[X_train_len::]

y_train_set = Y[0:X_train_len]
y_test_set = Y[X_train_len::]

# Print Train/Test set and their Lengths
"""print X_train_len
print X_test_len
print len(y_train_set)
print len(y_test_set)

print X_train_set
print X_test_set
print y_train_set
print y_test_set"""

#   Build the classifier
nn = Classifier(
    layers=[
        Layer("Rectifier", units=50),
        Layer("Softmax", name="OutputLayer", units=2)
    ],
    learning_rate=0.02,
    n_iter=100
)

# Fit the classifier
nn.fit(X_train_set, y_train_set)

# Form the prediction set for Test Data
y_pred = nn.predict(X_test_set)

# Calculate and Print the Accuracy
score = accuracy_score(y_test_set, y_pred)
print score


