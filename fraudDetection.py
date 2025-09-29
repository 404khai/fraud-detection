import pandas as pd

df = pd.read_csv("transactions.csv")

#first five rows
# print("\n First five rows \n", df.head())

#get missing values
# print("\n Missing values\n", df.isnull().sum())

print("\n Dataset Info. \n", df.info())
