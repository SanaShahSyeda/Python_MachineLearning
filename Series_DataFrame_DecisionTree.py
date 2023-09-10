# -*- coding: utf-8 -*-
"""
@author: Syeda Sana
"""


#Task1 -- series
import pandas as pd

a = ["Sana", "Sidra", "Hanood"]
myvar = pd.Series(a, index = ["46", "56", "01"])
print(myvar)

#Task2 -- DataFrame
data = {
        "cms_ids" : ["46","56","01"],
        "names" : ["Sana", "Sidra", "Hanood"]
        }
print(pd.DataFrame(data))

#ML- Decision Tree

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("shows.csv")
d = {"UK":0, "USA":1, "N":2}
data["Nationality"] = data["Nationality"].map(d)
d = {"YES": 1, "NO": 0}
data['Go']= data['Go'].map(d)


features = ['Age', 'Experience', 'Rank', 'Nationality']

x = data[features]
y = data['Go']


decisionTree = DecisionTreeClassifier()
decisionTree = decisionTree.fit(x, y)

tree.plot_tree(decisionTree, feature_names=features)



# UCI ML repository--------iris.data 
import pandas as pd
data = pd.read_csv("iris.csv")
d = {"setosa":0 , "versicolor":1, "virginica":2}
data['species']= data['species'].map(d)

features = ['sepal_length','sepal_width', 'petal_length','petal_width']

x = data[x]
y = data['species']

decisionTree = DecisionTreeClassifier()
decisionTree = decisionTree.fit(x, y)

tree.plot_tree(decisionTree, feature_names=features)
