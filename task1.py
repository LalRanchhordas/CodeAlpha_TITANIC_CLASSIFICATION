import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Titanic Dataset
df = pd.read_csv("Titanic-Dataset.csv")
print("Dataset loaded successfully")

# Display the first few rows of the dataset
print (df.head())

df.describe()