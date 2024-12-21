import pandas as pd
import matplotlib.pyplot as plt

# Load the Titanic Dataset
df = pd.read_csv("Titanic-Dataset.csv") 
# Pie Chart: Gender Distribution
plt.figure(figsize=(8, 8))
gender_distribution = df['Sex'].value_counts()
plt.pie(gender_distribution, labels=gender_distribution.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('Gender Distribution')
plt.show()