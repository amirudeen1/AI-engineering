import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

path= ''
await download(path,"drug200.csv")
path="drug200.csv"

my_data = pd.read_csv("drug200.csv", delimiter=",")
# my_data[0:5]

# my_data.shape 

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values # To set the features
# X[0:5]

# some features in this dataset are categorical, such as Sex or BP. Unfortunately, 
# Sklearn Decision Trees does not handle categorical variables. We can still convert these features to numerical values using LabelEncoder to convert the categorical variable into numerical variables.

# from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

# X[0:5]
y = my_data["Drug"]
# y[0:5]

# from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# To ensure dimensions match
# print('Shape of X_trainset: {}'.format(X_trainset.shape), 'Shape of y_trainset: {}'.format(y_trainset.shape))
# print('Shape of X_testset: {}'.format(X_testset.shape), 'Shape of y_testset: {}'.format(y_testset.shape))

# Modelling
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
# drugTree # it shows the default parameters

drugTree.fit(X_trainset,y_trainset)

predTree = drugTree.predict(X_testset)

# print (predTree [0:5])
# print (y_testset [0:5])

# from sklearn import metrics
# import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
tree.plot_tree(drugTree)
plt.show()





