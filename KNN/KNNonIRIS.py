import numpy as np;
import pandas as pd;
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score as acc
dataset=pd.read_csv("iris.csv")
x=dataset[['sepal length','sepal width','petal length','petal width']]
y=dataset['class']
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2)
classifier=knn(n_neighbors=7)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)
conf_matrix = cm(y_test, y_pred)
print(conf_matrix)
accuracy = acc(y_test,y_pred)*100
print(accuracy)
classifier=knn(n_neighbors=15)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)
conf_matrix = cm(y_test, y_pred)
print(conf_matrix)
accuracy = acc(y_test,y_pred)*100
print(accuracy)