import pandas as pd

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")  

# Describe the dataset
print(df.describe())
