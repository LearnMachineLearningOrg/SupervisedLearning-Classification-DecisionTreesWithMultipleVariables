# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:29:06 2018

@author: venkata
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#This function is used to load CSV file from the 'data' directory 
#in the present working directly 
def loadCSV (fileName):
    scriptDirectory = os.path.dirname(__file__)
    dataDirectoryPath = "."
    dataDirectory = os.path.join(scriptDirectory, dataDirectoryPath)
    dataFilePath = os.path.join(dataDirectory, fileName)
    return pd.read_csv(dataFilePath)

#This funtion is used to preview the data in the given dataset
def previewData (dataSet):
    print(dataSet.head())
    print("\n")

#This function is used to check for missing values in a given dataSet
def checkForMissingValues (dataSet):
    print(dataSet.isnull().sum())
    print("\n")

#This function is used to check the statistics of a given dataSet
def getStatisticsOfData (dataSet):
    print("***** Datatype of each column in the data set: *****")
    dataSet.info()
    print("\n")
    print("***** Columns in the data set: *****")
    print(dataSet.columns.values)
    print("***** Details about the data set: *****")
    print(dataSet.describe())
    print("\n")
    print("***** Checking for any missing values in the data set: *****")
    checkForMissingValues(dataSet)
    print("\n")
    
#This funtion is used to handle the missing value in the features, in the 
#given examples
def handleMissingValues (features):
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer.fit(features)
    imputedFeatures = imputer.fit_transform(features)
    return imputedFeatures
 
# Importing the dataset
dataset = pd.read_csv('social_network_ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

#Preview the dataSet and look at the statistics of the dataSet
#Check for any missing values 
#so that we will know whether to handle the missing values or not
print("***** Preview the dataSet and look at the statistics of the dataSet *****")
previewData(dataSet)
getStatisticsOfData(dataSet)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix: ", cm)
print ("Classification Rate/Accuracy: ", ((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))*100, "%")
print ("Recall: Recall gives us an idea about when it’s actually yes, how often does it predict yes: ", cm[1][1]/(cm[1][0]+cm[1][1]))
print ("Precision: Precsion tells us about when it predicts yes, how often is it correct.: ", cm[1][1]/(cm[0][1]+cm[1][1]))
"""
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()