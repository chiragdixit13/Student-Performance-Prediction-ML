#student performance prediction

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd

# Load dataset

df=pd.read_csv("dataset.csv")

# Encode categorical columns

int_encoder=LabelEncoder()
gen_encoder=LabelEncoder()
df["Internet"]=int_encoder.fit_transform(df["Internet"])

df["Gender"]=gen_encoder.fit_transform(df["Gender"])

# Select features and target

X=df[["Hours", "Sleep","Attendance","Previous_Score","Internet","Gender"]]
y=df["Marks"]

# Split dataset

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=32)

# Train Linear Regression model

model=LinearRegression()
model.fit(X_train,y_train)
# prediction=model.predict([[7,6,90,80,1,1]])

# Evaluate model

prediction2=model.predict(X_test)
print(mean_absolute_error(y_test,prediction2))
print(mean_squared_error(y_test,prediction2))
print(r2_score(y_test,prediction2))
# print(prediction2)

