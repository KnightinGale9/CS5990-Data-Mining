# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# 11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

# defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

# reading the training data
df = pd.read_csv("weather_training.csv", sep=',', header=0)
X_training = np.array(df.values)[:, 1:-1].astype('f')
train = np.array(df.values)[:, -1].astype('f')
y_training=[]
for y in train:
    for cl in classes:
        if cl - 3 <= y <= cl + 3:
            y_training.append(cl)
            break

# reading the test data
# hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
df = pd.read_csv("weather_test.csv", sep=',', header=0)
X_test = np.array(df.values)[:, 1:-1].astype('f')
test = np.array(df.values)[:, -1].astype('f')
y_test = []
for y in test:
    for cl in classes:
        if cl - 3 <= y <= cl + 3:
            y_test.append(cl)
            break


# loop over the hyperparameter values (k, p, and w) ok KNN
# --> add your Python code here
highest_accuracy=0
for kV in k_values:
    for pV in p_values:
        for wV in w_values:
            # fitting the knn to the data
            # --> add your Python code here
            clf = KNeighborsClassifier(n_neighbors=kV, p=pV, weights=wV)
            clf = clf.fit(X_training, y_training)

            # make the KNN prediction for each test sample and start computing its accuracy
            # hint: to iterate over two collections simultaneously, use zip()
            # Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            # to make a prediction do: clf.predict([x_testSample])
            # the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            # to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            # --> add your Python code here
            correct = 0
            for (X_testSample,y_testSample) in zip(X_test,y_test):
                prediction = clf.predict([X_testSample])
                if(abs(prediction[0]) - abs(y_testSample))==0:
                    correct+=1

            # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            # with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            # --> add your Python code here
            if (correct / len(y_test) > highest_accuracy):
                highest_accuracy = correct / len(y_test)
                print(f'Highest KNN accuracy so far:{highest_accuracy}, Parameters: k={kV}, p={pV}, w={wV}')