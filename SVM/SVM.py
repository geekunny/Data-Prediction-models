import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score as acc
from sklearn import preprocessing as p
from sklearn.preprocessing import StandardScaler as ss
data=pd.read_csv("data1modify.csv")
for ind,value in enumerate(data['precip.(mm)']):
    if(value>=0.0 and value<=4.0):
        data['precip.(mm)'][ind]=1
    elif(value>4.0 and value<=8.0):
        data['precip.(mm)'][ind]=2
    elif(value>8.0):
        data['precip.(mm)'][ind]=3
data['humidity()']=data['humidity()'].fillna(0).astype('int64')
label=p.LabelEncoder()
precip=label.fit_transform(data['precip.(mm)'])
x=data[['temp(c)','pressure(mb)','humidity()','wind speed(mph)','wind speed(mph)','wind dir.']]
y=data['precip.(mm)']
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2)
scaler=ss()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
model=SVC(gamma='auto')
model.fit(x_train,y_train)
predictions=model.predict(x_test)
confm=cm(y_test,predictions)
print("\nConfusion Matrix\n",confm)
accuracy=acc(y_test,predictions)*100
print("\nAccuracy of the model: %.2f "%accuracy+'%')
