import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Dataframe display settings
desired_width = 410
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)

df = pd.read_csv('KNN_Project_Data')
#print(df.head())

seaborn.pairplot(df)
#plt.show()

# Fitting and transforming data so the data fits a standard scale
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

# Creating a new dataframe with the scaled data
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
# print(df_feat.head())

# Train, test split
X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Getting results from using k=1 to determine accuracy
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Calculating error rate to get the optimal number of neighbors for most accurate classification
# Error rate = average amount of when the predicted value does not equal the test value
error_rate = []
for i in range(1, 40):
    knn = knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Graphing the error rates to get a visual to choose the best neighbor amount
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
# plt.show()

# From error rate graph, 31 appears to be the best neighbor amount for least error.
knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
