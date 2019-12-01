# split data to train and test (for faster calculation, just use 1/10 data)
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='./')

# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
images = mnist.data
targets = mnist.target

# make the value of pixels from [0, 255] to [0, 1] for further process
X = mnist.data / 255.
Y = mnist.target

# split data to train and test (for faster calculation, just use 1/10 data)
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

test_it=LogisticRegression();
test_it.fit(X_train,Y_train)
#train

testresult=test_it.predict(X_test)
trainresult=test_it.predict(X_train)
#predict

train_accuracy=metrics.accuracy_score(trainresult,Y_train)
test_accuracy=metrics.accuracy_score(testresult,Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))