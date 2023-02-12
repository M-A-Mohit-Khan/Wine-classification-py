import numpy as np
import pandas as pd
# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets # for using built-in datasets
from sklearn import metrics # for checking the model accuracy
#To plot the graph embedded in the notebook
import matplotlib.pyplot as plt 
from pandas.plotting import parallel_coordinates
from sklearn.model_selection import train_test_split
# importing the necessary package to use the classification algorithm
from sklearn import svm #for Support Vector Machine (SVM) Algorithm
# importing the necessary package to use the classification algorithm
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
# importing the necessary package to use the classification algorithm
from sklearn.neighbors import KNeighborsClassifier # for K nearest neighbours
# importing the necessary package to use the classification algorithm
from sklearn.linear_model import LogisticRegression # for Logistic Regression algorithm
# importing the necessary package to use the classification algorithm
from sklearn.naive_bayes import GaussianNB



# iris = datasets.load_iris()
# print(iris.data)

from sklearn.datasets import load_wine
wineData = load_wine()


# print(wineData.keys())

# Let's print the description of wine dataset
# print(wineData.DESCR)

#Let's print the data (features matrix) of wine dataset
# print(wineData.data)

# Let's check the shape of features matrix
# print(wineData.data.shape)

# Let's print the feature names
# print(wineData.feature_names)

# Let's print the target vector
# print(wineData.target)

# Let's check the shape of target
# print(wineData.target.shape)

# #Let's print the target class/species names 
# print(wineData.target_names)

# Let's load the data (features matrix) into pandas DataFrame
wine_df = pd.DataFrame(wineData.data, columns=wineData.feature_names)
# print(wine_df)

# Let's add target label into pandas DataFrame
wine_df['recog'] = wineData.target

# replace the target values with class names
wine_df['recog'] = wine_df['recog'].replace([0, 1, 2], ['class_0', 'class_1', 'class_2'])
# print(wine_df)


# let's check number of samples for each class of wine
# print(wine_df.groupby('recog').size())

# let's visualise the number of samples for each class with count plot
# sns.countplot(x='recog', data=wine_df)
# plt.show()

# Return numerical summary of each attribute of wine
# print(wine_df.describe())

# corr() to calculate the correlation between variables
correlation_matrix = wine_df.corr().round(2)

# changing the figure size
# plt.figure(figsize = (9, 6))
# # "annot = True" to print the values inside the square
# sns.heatmap(data=correlation_matrix, annot=True)
# plt.show()

# Steps to remove redundant values
# Return a array filled with zeros
# mask = np.zeros_like(correlation_matrix)
# # Return the indices for the upper-triangle of array
# mask[np.triu_indices_from(mask)] = True
# # changing the figure size
# plt.figure(figsize = (9, 6))
# # "annot = True" to print the values inside the square
# sns.heatmap(data=correlation_matrix, annot=True, mask=mask)
# plt.show()

# let's create pairplot to visualise the data for each pair of attributes
# sns.pairplot(wine_df, hue="recog", height = 2, palette = 'colorblind')
# plt.show()


# parallel_coordinates(wine_df, "recog", color = ['blue', 'red', 'green'])
# plt.show()


X = wine_df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']]
# print(X)

y = wine_df['recog']
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 16)
# print("X_train shape: ", X_train.shape)
# print("X_test shape: ", X_test.shape)
# print("y_train shape: ", y_train.shape)
# print("y_test shape: ", y_test.shape)

# model_svm = svm.SVC() #select the algorithm
# model_svm.fit(X_train, y_train) #train the model with the training dataset
# y_prediction_svm = model_svm.predict(X_test) # pass the testing data to the trained model
# # checking the accuracy of the algorithm.
# # by comparing predicted output by the model and the actual output
# score_svm = metrics.accuracy_score(y_prediction_svm, y_test).round(4)
# print("----------------------------------")
# print('The accuracy of the SVM is: {}'.format(score_svm))
# print("----------------------------------")
# # save the accuracy score
# score = set()
# score.add(('SVM', score_svm))



# model_dt = DecisionTreeClassifier(random_state=4)
# model_dt.fit(X_train, y_train) #train the model with the training dataset
# y_prediction_dt = model_dt.predict(X_test) #pass the testing data to the trained model
# # checking the accuracy of the algorithm.
# # by comparing predicted output by the model and the actual output
# score_dt = metrics.accuracy_score(y_prediction_dt, y_test).round(4)
# print("---------------------------------")
# print('The accuracy of the DT is: {}'.format(score_dt))
# print("---------------------------------")
# # save the accuracy score
# score = set()
# score.add(('DT', score_dt))


# from sklearn.linear_model import LogisticRegression # for Logistic Regression algorithm
# model_knn = KNeighborsClassifier(n_neighbors=3) # 3 neighbours for putting the new data into a class
# model_knn.fit(X_train, y_train) #train the model with the training dataset
# y_prediction_knn = model_knn.predict(X_test) #pass the testing data to the trained model
# # checking the accuracy of the algorithm.
# # by comparing predicted output by the model and the actual output
# score_knn = metrics.accuracy_score(y_prediction_knn, y_test).round(4)
# print("----------------------------------")
# print('The accuracy of the KNN is: {}'.format(score_knn))
# print("----------------------------------")
# # save the accuracy score
# score = set()
# score.add(('KNN', score_knn))



# model_lr = LogisticRegression()
# model_lr.fit(X_train, y_train) #train the model with the training dataset
# y_prediction_lr = model_lr.predict(X_test) #pass the testing data to the trained model
# # checking the accuracy of the algorithm.
# # by comparing predicted output by the model and the actual output
# score_lr = metrics.accuracy_score(y_prediction_lr, y_test).round(4)
# print("---------------------------------")
# print('The accuracy of the LR is: {}'.format(score_lr))
# print("---------------------------------")
# # save the accuracy score
# score = set()
# score.add(('LR', score_lr))



model_nb = GaussianNB()
model_nb.fit(X_train, y_train) #train the model with the training dataset
y_prediction_nb = model_nb.predict(X_test) #pass the testing data to the trained model
# checking the accuracy of the algorithm.
# by comparing predicted output by the model and the actual output
score_nb = metrics.accuracy_score(y_prediction_nb, y_test).round(4)
print("---------------------------------")
print('The accuracy of the NB is: {}'.format(score_nb))
print("---------------------------------")
# save the accuracy score
score = set()
score.add(('NB', score_nb))



