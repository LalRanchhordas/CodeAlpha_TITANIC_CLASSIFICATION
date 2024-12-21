import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic Dataset
df = pd.read_csv("Titanic-Dataset.csv")  # Ensure the dataset path is correct

# Histogram: Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'].dropna(), bins=30, kde=True, color='blue', alpha=0.7)  # Added transparency for clarity
plt.title('Age Distribution', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid for better visualization
plt.show()
