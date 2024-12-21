import pandas as pd  # Import pandas
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic Dataset
df = pd.read_csv("Titanic-Dataset.csv")  # Ensure the file path is correct

# Scatter Plot: Age vs. Fare
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)  # Ensure column names match the dataset
plt.title('Scatter Plot: Age vs. Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived')
plt.show()
