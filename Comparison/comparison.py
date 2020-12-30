import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as ttl
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.metrics import precision_score,recall_score,f1_score
data=pd.read_csv("data1modify.csv")
for ind,value in enumerate(data['precip.(mm)']):
    if(value>=0.0 and value<=4.0):
        data['precip.(mm)'][ind]=1
    elif(value>4.0 and value<=8.0):
        data['precip.(mm)'][ind]=2
    elif(value>8.0):
        data['precip.(mm)'][ind]=3
data['humidity()']=data['humidity()'].fillna(0).astype('int64')
X=data[['temp(c)','pressure(mb)','humidity()','wind speed(mph)','wind speed(mph)','wind dir.']]
y=data['precip.(mm)']
names=["KNN","SVM","Decision Tree",
       "Neural Network","Naive Bayesian"]
classifiers=[
    KNN(3),
    SVC(kernel="linear",C=0.025),
    DTC(max_depth=5),
    MLP(alpha=1,max_iter=1000),
    NB()]
x_train,x_test,y_train,y_test=ttl(X,y,test_size=0.3,random_state=1)
model_cols=[]
comparison=pd.DataFrame(columns=model_cols)
index=0
for name,clf in zip(names,classifiers):
    clf.fit(x_train,y_train)
    comparison.loc[index,'Classifiers']=name
    comparison.loc[index,'Train Accuracy']=clf.score(x_train,y_train)
    comparison.loc[index,'Test Accuracy']=clf.score(x_test,y_test)
    comparison.loc[index,'Precision']=precision_score(y_test,clf.predict(x_test),average='macro')
    comparison.loc[index,'Recall']=recall_score(y_test,clf.predict(x_test),average='macro')
    comparison.loc[index,'F1 Score'] = f1_score(y_test,clf.predict(x_test),average='macro')
    index+=1
comparison
