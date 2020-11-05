import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score as acc
from sklearn import preprocessing as p

data=pd.read_csv("Rainfall.csv")
for ind,value in enumerate(data['precip.(mm)']):
    if(value>=0.0 and value<=4.0):
        data['precip.(mm)'][ind]='Low Rainfall'
    elif(value>4.0 and value<=8.0):
        data['precip.(mm)'][ind]='Moderate Rainfall'
    elif(value>8.0):
        data['precip.(mm)'][ind]='High Rainfall'
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

mc=mlp(hidden_layer_sizes=(10,10,10),activation='logistic',solver='sgd',learning_rate_init=0.9,max_iter=1000)
mc.fit(x_train,y_train)

predictions=mc.predict(x_test)

confm=cm(y_test,predictions)
print("\nConfusion Matrix\n",confm)
accuracy=acc(y_test,predictions)*100
print("\nAccuracy of the model: %.2f "%accuracy+'%')
