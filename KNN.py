# !pip install scikit-learn==0.23.1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
%matplotlib inline

df = pd.read_csv('')
# df.head()

# Visualization and analyse the number of counts of each class
# df['custcat'].value_counts()
# df.hist(column='income', bins=50)

# df.columns

## Convert to np df
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
# X[0:5]

# y = df['custcat'].values -> labels
# y[0:5]

## Normalize data for 0 mean and 1 var
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# X[0:5]

## Train test split
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape, y_train.shape)
# print ('Test set:', X_test.shape, y_test.shape)

# from sklearn.neighbors import KNeighborsClassifier

k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

yhat = neigh.predict(X_test)

## Accuracy
# from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

## To find the best k value to use
# Ks = 10
# mean_acc = np.zeros((Ks-1))
# std_acc = np.zeros((Ks-1))

# for n in range(1,Ks):
    
    #Train Model and Predict  
    # neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    # yhat=neigh.predict(X_test)
    # mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    # std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

# mean_acc


## For visualization
# plt.plot(range(1,Ks),mean_acc,'g')
# plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
# plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
# plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
# plt.ylabel('Accuracy ')
# plt.xlabel('Number of Neighbors (K)')
# plt.tight_layout()
# plt.show()
