import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymongo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

client = pymongo.MongoClient('mongodb://localhost:27017')
db = client["student_marks"]
collection = db['Student_scores']
documents = collection.find({})
all_data=[]
for i in documents:
   dict_data = {}
   dict_data['Hours'] = i['Hours']
   dict_data['Scores'] = i['Scores']
   all_data.append(dict_data)

df = pd.DataFrame(all_data)
print(df)

# identify a data types:
print(df.dtypes)

# identifyn a null values :
print(df.isnull().sum())

# remove duplicates values :
df = df.drop_duplicates()
print(df)

# statistical measures :
print(df.describe())

# checking IQR (Inter Quantile Range) :

# IQR = Q3 - Q1
IQR = df.Hours.quantile(0.75) - df.Hours.quantile(0.25)
print(IQR)

# Upper_Threshold
Upper_Threshold = df.Hours.quantile(0.75) + (1.5 * IQR)
print(Upper_Threshold)

# Lower_Threshold
Lower_Threshold = df.Hours.quantile(0.25) - (1.5 * IQR)
print(Lower_Threshold)

# data visualization :
df.plot(x="Hours",y="Scores",style="o")
plt.title("Hours VS Scores")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# co-relation matrix :
print(df.corr())

# train and test split :
X = df[["Hours"]]
y = df["Scores"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
LR = LinearRegression()
LR.fit(X_train,y_train)

# y = m*X + c
print(LR.intercept_) # c- Values
print(LR.coef_) # m- Values

# prediction values :
# 8 Hours Stuides
print(LR.predict([[8]]))

# Compare Actual and Predicted Values :
y_pred = LR.predict(X_test)
print(y_pred)

df = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
print(df)

# evaluation metrics :
from sklearn import metrics
print("R2-Score: ",metrics.r2_score(y_test,y_pred))