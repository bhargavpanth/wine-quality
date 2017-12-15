from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/winequality-white.csv', header=0, sep=';')

X = df[list(df.columns)[:-1]]
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print 'Score:', model.score(X_test, y_test)
print 'RMSE:', mean_squared_error(y_predict, y_test) ** 0.5
