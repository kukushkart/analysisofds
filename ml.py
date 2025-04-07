import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import kagglehub
from sklearn.preprocessing import MinMaxScaler


path = kagglehub.dataset_download("amulyas/penguin-size-dataset")

print("Path to dataset files:", path)

df = pd.read_csv(f"{path}/penguin_size.csv")

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())


numeric_cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())


df['sex'] = df['sex'].fillna(df['sex'].mode()[0])


plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_cols])
plt.title("Distribution of numerical features")
plt.show()


for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= (Q1 - 1.5*IQR)) & (df[col] <= (Q3 + 1.5*IQR))]

df = pd.get_dummies(df, columns=['island', 'species'])
df['sex'] = df['sex'].map({'MALE': 0, 'FEMALE': 1})

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

corr_matrix = df.corr(method='pearson')
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation matrix")
plt.show()

df[numeric_cols].hist(figsize=(10, 8))
plt.suptitle("Distribution of numerical features")
plt.show()

sns.pairplot(df, vars=numeric_cols, hue='sex', diag_kind='kde')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='species_Adelie', y='body_mass_g', data=df)
plt.title("Body weight by penguin species")
plt.show()

for col in numeric_cols:
    stat, p = stats.shapiro(df[col])
    print(f"{col}: p-value = {p:.3f}")


male = df[df['sex'] == 0]['body_mass_g']
female = df[df['sex'] == 1]['body_mass_g']
t_stat, p_val = stats.ttest_ind(male, female)
print(f"p-value: {p_val:.4f}")