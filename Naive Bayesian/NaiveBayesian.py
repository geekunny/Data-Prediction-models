import pandas as pd
import numpy as np
store=dict({})
dataset=pd.read_csv("dataset.csv")
for i in dataset.columns:
    store.update({i:np.array(dataset[i])})
print("Enter the data according to you, We will predict")
age_i=input("Enter age(youth/middle aged/senior): ")
inc_i=input("Enter income(low/medium/high): ")
std_i=input("Enter student(yes/no): ")
cr_i=input("Enter credit rating(fair/excellent)")
input_l=[age_i,inc_i,std_i,cr_i]
rows=len(store['RID'])
count=0
for i in store['Class: buys computer']:
    if(i=='yes'):
        count+=1
no=rows-count
p_yes=count/rows
p_no=no/rows
Yes_A=({})
No_A=({})
del store['RID']
l=store['Class: buys computer']
del store['Class: buys computer']
for attr in store: #attr=income
    for val in set(store[attr]): #val=middle aged
        count1,count2=0,0
        for ind,value in enumerate(store[attr]):#value=youth,ind=10
            if(value==val):
                if(l[ind]=="yes"): #l[10]="yes"
                    count1+=1
                elif(l[ind]=="no"): #l[10]="no"
                    count2+=1
            t=count1+count2 #t=5
        if t!=0:
            Yes_A[val]=count1/count #yes_a=2/9 #yes_A[youth:2/9]
            No_A[val]=count2/no #no_a=3/5 #no_A[youth:3/5]
product_yes=1
product_no=1
for j in input_l:
    for i in Yes_A:
        if(i==j):
            product_yes*=Yes_A[i]
    for k in No_A:
        if(j==k):
            product_no*=No_A[k]
predict_yes=product_yes*p_yes
predict_no=product_no*p_no
if(predict_yes>predict_no):
    print("Prediction: Customer buys the computer")
else:
    print("Prediction: Customer does not buys the computer")