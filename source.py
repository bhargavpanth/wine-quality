
# coding: utf-8

# In[110]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# In[6]:


# red_wine_df = pd.read_csv('winequalityred.csv',sep=';')
white_wine_df = pd.read_csv('data/winequality-white.csv',sep=';')

# #Red Wine Attributes Analysis(In relationship with target output and against each other)

# In[7]:


# red_wine_df.head()

# In[8]:


# sns.distplot(red_wine_df['quality'])
# plt.show()

# In[9]:


# a=red_wine_df['quality']
# b=red_wine_df['fixed acidity']
# print (np.corrcoef(a,b))

# sns.boxplot(x='quality', y='fixed acidity', data=red_wine_df)
# plt.show()
# sns.barplot(x='quality', y='fixed acidity', data=red_wine_df)
# plt.show()
# sns.swarmplot(x='quality', y='fixed acidity', data=red_wine_df)
# plt.show()


# In[10]:


# a=red_wine_df['quality']
# b=red_wine_df['volatile acidity']
# print (np.corrcoef(a,b))

# sns.boxplot(x='quality', y='volatile acidity', data=red_wine_df)
# plt.show()
# sns.barplot(x='quality', y='volatile acidity', data=red_wine_df)
# plt.show()
# sns.swarmplot(x='quality', y='volatile acidity', data=red_wine_df)
# plt.show()

# In[11]:


# a=red_wine_df['quality']
# b=red_wine_df['citric acid']
# print (np.corrcoef(a,b))

# sns.boxplot(x='quality', y='citric acid',data=red_wine_df)
# plt.show()
# sns.barplot(x='quality', y='citric acid', data=red_wine_df)
# plt.show()
# sns.swarmplot(x='quality', y='citric acid', data=red_wine_df)
# plt.show()

# In[12]:


# a=red_wine_df['quality']
# b=red_wine_df['residual sugar']
# print (np.corrcoef(a,b))

# sns.boxplot(x='quality', y='residual sugar', data=red_wine_df)
# plt.show()
# sns.barplot(x='quality', y='residual sugar', data=red_wine_df)
# plt.show()
# sns.swarmplot(x='quality', y='residual sugar', data=red_wine_df)
# plt.show()

# In[13]:


# a=red_wine_df['quality']
# b=red_wine_df['chlorides']
# print (np.corrcoef(a,b))

# sns.boxplot(x='quality', y='chlorides', data=red_wine_df)
# plt.show()
# sns.barplot(x='quality', y='chlorides', data=red_wine_df)
# plt.show()
# sns.swarmplot(x='quality', y='chlorides', data=red_wine_df)
# plt.show()

# In[14]:


# a=red_wine_df['quality']
# b=red_wine_df['free sulfur dioxide']
# print (np.corrcoef(a,b))

# sns.boxplot(x='quality', y='free sulfur dioxide', data=red_wine_df)
# plt.show()
# sns.barplot(x='quality', y='free sulfur dioxide', data=red_wine_df)
# plt.show()
# sns.swarmplot(x='quality', y='free sulfur dioxide', data=red_wine_df)
# plt.show()

# In[15]:


# a=red_wine_df['quality']
# b=red_wine_df['total sulfur dioxide']
# print (np.corrcoef(a,b))

# sns.boxplot(x='quality', y='total sulfur dioxide', data=red_wine_df)
# plt.show()
# sns.barplot(x='quality', y='total sulfur dioxide', data=red_wine_df)
# plt.show()
# sns.swarmplot(x='quality', y='total sulfur dioxide', data=red_wine_df)
# plt.show()

# In[16]:


# a=red_wine_df['quality']
# b=red_wine_df['density']
# print (np.corrcoef(a,b))

# sns.boxplot(x='quality', y='density', data=red_wine_df)
# plt.show()
# sns.barplot(x='quality', y='density', data=red_wine_df)
# plt.show()
# sns.swarmplot(x='quality', y='density', data=red_wine_df)
# plt.show()

# In[17]:


# a=red_wine_df['quality']
# b=red_wine_df['pH']
# print (np.corrcoef(a,b))

# sns.boxplot(x='quality', y='pH', data=red_wine_df)
# plt.show()
# sns.barplot(x='quality', y='pH', data=red_wine_df)
# plt.show()
# sns.swarmplot(x='quality', y='pH', data=red_wine_df)
# plt.show()

# In[18]:


# a=red_wine_df['quality']
# b=red_wine_df['sulphates']
# print (np.corrcoef(a,b))

# sns.boxplot(x='quality', y='sulphates', data=red_wine_df)
# plt.show()
# sns.barplot(x='quality', y='sulphates', data=red_wine_df)
# plt.show()
# sns.swarmplot(x='quality', y='sulphates', data=red_wine_df)
# plt.show()

# In[19]:


# a=red_wine_df['quality']
# b=red_wine_df['alcohol']
# print (np.corrcoef(a,b))

# sns.boxplot(x='quality', y='alcohol', data=red_wine_df)
# plt.show()
# sns.barplot(x='quality', y='alcohol', data=red_wine_df)
# plt.show()
# sns.swarmplot(x='quality', y='alcohol', data=red_wine_df)
# plt.show()

# In[20]:


# a=red_wine_df['free sulfur dioxide']
# b=red_wine_df['total sulfur dioxide']
# print (np.corrcoef(a,b))

# sns.swarmplot(x='free sulfur dioxide', y='total sulfur dioxide', data=red_wine_df)
# plt.show()

# In[21]:


# a=red_wine_df['fixed acidity']
# b=red_wine_df['citric acid']
# print (np.corrcoef(a,b))

# sns.swarmplot(x='fixed acidity', y='citric acid', data=red_wine_df)
# plt.show()

# In[22]:


# a=red_wine_df['fixed acidity']
# b=red_wine_df['pH']
# print (np.corrcoef(a,b))

# sns.swarmplot(x='fixed acidity', y='pH', data=red_wine_df)
# plt.show()

# In[23]:


# a=red_wine_df['alcohol']
# b=red_wine_df['density']
# print (np.corrcoef(a,b))

# sns.swarmplot(x='alcohol', y='density', data=red_wine_df)
# plt.show()

# In[24]:


# new_red_df = red_wine_df.drop(['citric acid', 'pH','free sulfur dioxide', 'density','residual sugar'], axis=1,inplace=False)
# red_wine_df.head()

# #White Wine Attributes Analysis(In relationship with target output and against each other)

# In[25]:


# white_wine_df.head()

# In[26]:


sns.distplot(white_wine_df['quality'])
plt.show()

# In[27]:


a=white_wine_df['quality']
b=white_wine_df['fixed acidity']
print (np.corrcoef(a,b))

sns.boxplot(x='quality', y='fixed acidity', data=white_wine_df)
plt.show()
sns.barplot(x='quality', y='fixed acidity', data=white_wine_df)
plt.show()
sns.swarmplot(x='quality', y='fixed acidity', data=white_wine_df)
plt.show()


# In[28]:


a=white_wine_df['quality']
b=white_wine_df['volatile acidity']
print (np.corrcoef(a,b))

sns.boxplot(x='quality', y='volatile acidity', data=white_wine_df)
plt.show()
sns.barplot(x='quality', y='volatile acidity', data=white_wine_df)
plt.show()
sns.swarmplot(x='quality', y='volatile acidity', data=white_wine_df)
plt.show()

# In[29]:


a=white_wine_df['quality']
b=white_wine_df['citric acid']
print (np.corrcoef(a,b))

sns.boxplot(x='quality', y='citric acid',data=white_wine_df)
plt.show()
sns.barplot(x='quality', y='citric acid', data=white_wine_df)
plt.show()
sns.swarmplot(x='quality', y='citric acid', data=white_wine_df)
plt.show()

# In[30]:


a=white_wine_df['quality']
b=white_wine_df['residual sugar']
print (np.corrcoef(a,b))

sns.boxplot(x='quality', y='residual sugar', data=white_wine_df)
plt.show()
sns.barplot(x='quality', y='residual sugar', data=white_wine_df)
plt.show()
sns.swarmplot(x='quality', y='residual sugar', data=white_wine_df)
plt.show()

# In[31]:


a=white_wine_df['quality']
b=white_wine_df['chlorides']
print (np.corrcoef(a,b))

sns.boxplot(x='quality', y='chlorides', data=white_wine_df)
plt.show()
sns.barplot(x='quality', y='chlorides', data=white_wine_df)
plt.show()
sns.swarmplot(x='quality', y='chlorides', data=white_wine_df)
plt.show()

# In[32]:


a=white_wine_df['quality']
b=white_wine_df['free sulfur dioxide']
print (np.corrcoef(a,b))

sns.boxplot(x='quality', y='free sulfur dioxide', data=white_wine_df)
plt.show()
sns.barplot(x='quality', y='free sulfur dioxide', data=white_wine_df)
plt.show()
sns.swarmplot(x='quality', y='free sulfur dioxide', data=white_wine_df)
plt.show()

# In[33]:


a=white_wine_df['quality']
b=white_wine_df['total sulfur dioxide']
print (np.corrcoef(a,b))

sns.boxplot(x='quality', y='total sulfur dioxide', data=white_wine_df)
plt.show()
sns.barplot(x='quality', y='total sulfur dioxide', data=white_wine_df)
plt.show()
sns.swarmplot(x='quality', y='total sulfur dioxide', data=white_wine_df)
plt.show()

# In[34]:


a=white_wine_df['quality']
b=white_wine_df['density']
print (np.corrcoef(a,b))

sns.boxplot(x='quality', y='density', data=white_wine_df)
plt.show()
sns.barplot(x='quality', y='density', data=white_wine_df)
plt.show()
sns.swarmplot(x='quality', y='density', data=white_wine_df)
plt.show()

# In[35]:


a=white_wine_df['quality']
b=white_wine_df['pH']
print (np.corrcoef(a,b))

sns.boxplot(x='quality', y='pH', data=white_wine_df)
plt.show()
sns.barplot(x='quality', y='pH', data=white_wine_df)
plt.show()
sns.swarmplot(x='quality', y='pH', data=white_wine_df)
plt.show()

# In[36]:


a=white_wine_df['quality']
b=white_wine_df['sulphates']
print (np.corrcoef(a,b))

sns.boxplot(x='quality', y='sulphates', data=white_wine_df)
plt.show()
sns.barplot(x='quality', y='sulphates', data=white_wine_df)
plt.show()
sns.swarmplot(x='quality', y='sulphates', data=white_wine_df)
plt.show()

# In[37]:


a=white_wine_df['quality']
b=white_wine_df['alcohol']
print (np.corrcoef(a,b))

sns.boxplot(x='quality', y='alcohol', data=white_wine_df)
plt.show()
sns.barplot(x='quality', y='alcohol', data=white_wine_df)
plt.show()
sns.swarmplot(x='quality', y='alcohol', data=white_wine_df)
plt.show()

# In[38]:


a=white_wine_df['free sulfur dioxide']
b=white_wine_df['total sulfur dioxide']
print (np.corrcoef(a,b))

sns.swarmplot(x='free sulfur dioxide', y='total sulfur dioxide', data=white_wine_df)
plt.show()

# In[39]:


a=white_wine_df['fixed acidity']
b=white_wine_df['citric acid']
print (np.corrcoef(a,b))

sns.swarmplot(x='fixed acidity', y='citric acid', data=white_wine_df)
plt.show()

# In[40]:


a=white_wine_df['fixed acidity']
b=white_wine_df['pH']
print (np.corrcoef(a,b))

sns.swarmplot(x='fixed acidity', y='pH', data=white_wine_df)
plt.show()

# In[41]:


a=white_wine_df['alcohol']
b=white_wine_df['density']
print (np.corrcoef(a,b))

sns.swarmplot(x='alcohol', y='density', data=white_wine_df)
plt.show()

# In[81]:


new_white_df = white_wine_df.drop(['pH','free sulfur dioxide', 'density',
                               'residual sugar', 'citric acid'], axis=1,inplace=False)
new_white_df.head()

# #Separating 15 percent of the dataset as test set

# In[43]:


# X_train = red_wine_df.drop(['quality'], axis=1, inplace=False)
# y_train = red_wine_df['quality']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.15, random_state=0)

# #Trying 4 different models for regression 

# In[44]:


model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_, model.intercept_)

# In[99]:


y_pred = model.predict(X_test)
y_p=[]
for i in y_pred:
    y_p.append(int(i))
print ("Accuracy is:", np.mean(y_p == y_test))

print('Mean square error  is:', mean_squared_error(y_pred, y_test))
print('Mean absolute error is:', mean_absolute_error(y_pred, y_test))

# In[68]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.15, random_state=0)

for i in [3,5,7,10,15,20]:
    model = KNeighborsRegressor(n_neighbors=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print( i, mean_absolute_error(y_pred, y_val))     

# In[100]:


model = KNeighborsRegressor(n_neighbors=7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Mean square error  is:', mean_squared_error(y_pred, y_test))
print('Mean absolute error is:', mean_absolute_error(y_pred, y_test))

# In[74]:


for samples in [3,5,7,10,15,20]:
    for leaf in [1,3,5,7,10,15]:
        model = DecisionTreeRegressor(min_samples_split=samples, min_samples_leaf=leaf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print("When samples split is:{0} and min leaf is{1}".format(samples, leaf),
              mean_absolute_error(y_pred, y_val))

# In[101]:


model = DecisionTreeRegressor(min_samples_split=7, min_samples_leaf=15)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Mean square error  is:', mean_squared_error(y_pred, y_test))
print('Mean absolute error is:', mean_absolute_error(y_pred, y_test))

# #Decision tree outperforming other 3 models 

# In[111]:


model = SVR()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Mean square error  is:', mean_squared_error(y_pred, y_test))
print('Mean absolute error is:', mean_absolute_error(y_pred, y_test))

# #Using decision tree to model and evaluate all 4 of my datasets

# In[102]:


X_train = new_red_df.drop(['quality'], axis=1, inplace=False)
y_train = new_red_df['quality']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.15, random_state=0)
model = DecisionTreeRegressor(min_samples_split=7, min_samples_leaf=15)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Mean square error  is:', mean_squared_error(y_pred, y_test))
print('Mean absolute error is:', mean_absolute_error(y_pred, y_test))

# In[108]:


X_train = white_wine_df.drop(['quality'], axis=1, inplace=False)
y_train = white_wine_df['quality']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.15, random_state=0)
model = DecisionTreeRegressor(min_samples_split=7, min_samples_leaf=15)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Mean square error  is:', mean_squared_error(y_pred, y_test))
print('Mean absolute error is:', mean_absolute_error(y_pred, y_test))

# In[112]:


X_train = new_white_df.drop(['quality'], axis=1, inplace=False)
y_train = new_white_df['quality']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.15, random_state=0)
model = DecisionTreeRegressor(min_samples_split=7, min_samples_leaf=15)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Mean square error  is:', mean_squared_error(y_pred, y_test))
print('Mean absolute error is:', mean_absolute_error(y_pred, y_test))
