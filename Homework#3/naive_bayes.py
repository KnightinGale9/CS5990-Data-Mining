#-------------------------------------------------------------------------
# AUTHOR: Zhong Ooi
# FILENAME: naive_bayes.py
# SPECIFICATION: navie bayes calculation on weather data
# FOR: CS 5990- Assignment #3
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]
scaler = StandardScaler()

#reading the training data
#--> add your Python code here
# reading the training data
df = pd.read_csv("weather_training.csv", sep=',', header=0)
X_training = np.array(df.values)[:, 1:-1].astype('f')
X_training = scaler.fit_transform(X_training)
#update the training class values according to the discretization (11 values only)
#--> add your Python code here
train = np.array(df.values)[:, -1].astype('f')
y_training=[]
for y in train:
    for cl in classes:
        if cl - 3 <= y <= cl + 3:
            y_training.append(cl)
            break
#reading the test data
#--> add your Python code here
df = pd.read_csv("weather_test.csv", sep=',', header=0)
X_test = np.array(df.values)[:, 1:-1].astype('f')
X_test = scaler.transform(X_test)
#update the test class values according to the discretization (11 values only)
#--> add your Python code here
test = np.array(df.values)[:, -1].astype('f')
y_test = []
for y in test:
    for cl in classes:
        if cl - 3 <= y <= cl + 3:
            y_test.append(cl)
            break
#fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(X_training, y_training)

#make the naive_bayes prediction for each test sample and start computing its accuracy
#the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
#to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
#--> add your Python code here
accuracy=0
for (X_testSample, y_testSample) in zip(X_test, y_test):
    prediction = clf.predict([X_testSample])
    difference = (abs(prediction - y_testSample)) / y_testSample
    if difference<=0.15:
        accuracy += 1
accuracy=accuracy/len(y_test)
#print the naive_bayes accuracyy
#--> add your Python code here
print("naive_bayes accuracy: " + str(accuracy))



