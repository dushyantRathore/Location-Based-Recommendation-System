import numpy as np
import pandas as pd
import scipy
from sknn.mlp import Classifier, Regressor, Layer

#f = pd.read_csv('Dataset.csv', delimiter = '\t')
#df = pd.DataFrame(f)

X = np.loadtxt('Dataset.txt', usecols=[1, 2])
y = np.loadtxt('Dataset.txt', usecols=[3])

#print X
#print y

nn = Classifier(
        layers=[
                Layer("Rectifier", units=100),
                Layer("Softmax", name="OutputLayer", units=2)
            ],
        learning_rate = 0.02,
        n_iter=100
    )

#print(y)

nn.fit(X,y)

parameters = nn.get_parameters()

y_answer = nn.predict(np.array([[2, 25]]))

print(parameters);

print(y_answer);
