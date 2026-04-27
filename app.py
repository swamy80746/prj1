import pandas as pd
import numpy as np
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('ex1.csv')

df.drop('EmpID',axis=1,inplace=True)

#----------Age Column Cleaning-----------------------
df['Age'].replace(['thirty','?'],np.nan,inplace=True)
df['Age']=pd.to_numeric(df['Age'],errors='coerce')
df['Age']=df['Age'].fillna(df['Age'].mean().round(2))

#---------Experience Column Cleaning--------------------
df['Experience'].replace(['?','five'],np.nan)
df['Experience']=pd.to_numeric(df['Experience'],errors='coerce')
df['Experience']=df['Experience'].fillna(df['Experience'].mean().round(2))

#-------Department columns cleanig and convert into int--------------
le=LabelEncoder()
df['Department']=le.fit_transform(df['Experience'])

#---------Salary Column Cleaning--------------------
df['Salary'].replace(['?','sixty thousand'],np.nan)
df['Salary']=pd.to_numeric(df['Salary'],errors='coerce')
df['Salary']=df['Salary'].fillna(df['Salary'].mean().round(1))

model=LinearRegression()


xtrain,xtest,ytrain,ytest=train_test_split(df[['Age','Experience','Department']],df['Salary'],train_size=0.8)

model.fit(xtrain,ytrain)



age=st.number_input('Enter Your Age ')
experience=st.number_input('Enter Your Experience')
department=st.number_input('Enter Your Department')
pred=model.predict([[age,experience,department]])
if st.button('click on button'):
    
    st.text(f"your predict salary is {pred}")

