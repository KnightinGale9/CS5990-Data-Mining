# -------------------------------------------------------------------------
# AUTHOR: Zhong Ooi
# FILENAME: decision_tree.py
# SPECIFICATION: Transforming training and testing data to usable format for sklear decision tree.
#               Then running accuracy evaluation over the two training sets with a test set.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 1 hour and 30 mins
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)  # reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:, 1:]  # creating a training matrix without the id (NumPy library)
    # transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    # Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    # be converted to a float.
    data_training[data_training == 'Yes'] = 1
    data_training[data_training == 'No'] = 0
    for instance in data_training:
        temp = []
        temp.append(int(instance[0]))
        if (instance[1] == 'Single'):
            temp.append(1)
        else:
            temp.append(0)
        if (instance[1] == 'Divorced'):
            temp.append(1)
        else:
            temp.append(0)
        if (instance[1] == 'Married'):
            temp.append(1)
        else:
            temp.append(0)
        temp.append(float(str(instance[2]).replace("k", "")))
        temp.append(int(instance[3]))
        X.append(temp[:-1])
        # transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
        # --> add your Python code here
        Y.append(temp[-1])
    # print(ds)
    # for i,j in zip(X,Y):
    #     print("feature:",i,"class",j)
    model_accuracy = []
    # loop your training and test tasks 10 times here
    for i in range(10):
        # fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf = clf.fit(X, Y)

        # plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'],
                       class_names=['Yes', 'No'], filled=True, rounded=True)
        plt.show()

        # read the test data and add this data to data_test NumPy
        # --> add your Python code here
        df_test = pd.read_csv('cheat_test.csv', sep=',',
                              header=0)  # reading a dataset eliminating the header (Pandas library)
        data_test = np.array(df_test.values)[:, 1:]  # creating a training matrix without the id (NumPy library)
        data_test[data_test == 'Yes'] = 1
        data_test[data_test == 'No'] = 0
        run_accuracy = 0
        for data in data_test:
            temp = []
            temp.append(int(data[0]))
            if (data[1] == 'Single'):
                temp.append(1)
            else:
                temp.append(0)
            if (data[1] == 'Divorced'):
                temp.append(1)
            else:
                temp.append(0)
            if (data[1] == 'Married'):
                temp.append(1)
            else:
                temp.append(0)
            temp.append(float(str(data[2]).replace("k", "")))
            temp.append(int(data[3]))
            # transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            # class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            class_predicted = clf.predict([temp[:-1]])[0]
            # print(temp,class_predicted)
            # compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            # --> add your Python code here
            if (class_predicted == temp[-1]):
                run_accuracy += 1
        # find the average accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here
        model_accuracy.append(run_accuracy/len(data_test))
    # print the accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    # --> add your Python code here
    accuracy=0
    for acc in model_accuracy:
        accuracy+=acc
    print(f'Final accuracy from averaging 10 test run when training on {ds}: {accuracy/len(model_accuracy)} ({model_accuracy})')