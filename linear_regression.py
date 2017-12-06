from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data/winequality-white.csv', header=0, sep=';')

# X = df[list(df.columns)[:-1]]
X = df['alcohol'].reshape(-1,1)
y = df['quality']

plt.scatter(X,y)
plt.xlabel('Alcohol')
plt.ylabel('Quality') 
plt.show()

# print (X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print 'Score:', model.score(X_test, y_test)
print 'RMSE:', mean_squared_error(y_predict, y_test) ** 0.5

plt.figure(figsize=(20,10))
plt.title('Figure 3: Predictions vs true targets')
plt.xlabel('True targets')
plt.ylabel('Predictions')     
plt.xlim(0,10)
plt.ylim(0, 10)
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(y_test,y_predict,s=80,marker='o')
plt.plot(np.arange(10),np.arange(10),color='r')
plt.show()