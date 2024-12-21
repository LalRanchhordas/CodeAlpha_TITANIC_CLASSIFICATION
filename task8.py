import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic Dataset
df = pd.read_csv("Titanic-Dataset.csv")  # Ensure the dataset is in the correct directory or update the path

# Bar Graph: Survival Count
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=df, palette='viridis')  # Added a palette for improved visuals
plt.title('Survival Count', fontsize=16)
plt.xlabel('Survived (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.show()
