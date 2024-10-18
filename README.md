# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
 step 1: import the required libraries.
 step 2: Upload and read the dataset.
 step 3: Check for any null values using the isnull() function.
 step 4: From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
 step 5: Find the accuracy of the model and predict the required values by importing the required module from sklearn.
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Harrish Venkat V
RegisterNumber: 212223240049
*/

import pandas as pd
data = pd.read_csv('Employee.csv')
data.head()
data.isnull().sum()
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## Accuracy
![image](https://github.com/user-attachments/assets/d75e9db9-5da6-4467-b811-305647c37e78)

## Predicted
![image](https://github.com/user-attachments/assets/4112e294-5f25-4b00-ad52-84609a7b9049)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
