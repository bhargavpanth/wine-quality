
# coding: utf-8

# # Wine Quality
# 
# ## Notebook by [WenyiXu](https://github.com/xuwenyihust)
# 
# ### Import libraries

# In[1]:


# %matplotlib inline

import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn.cross_validation

# ### Load the data

# In[2]:


df = pd.read_csv("data/winequality-white.csv", sep=';')
# df.head()

# In[4]:


print(df.shape)

# Separate the dataset into **feature matrix X** & **respoinse vector y**.

# In[5]:


X_df = df.iloc[:,:-1]
X_df.head()

# In[6]:


X = X_df.as_matrix()
print(X[:3])

# In[7]:


y_df = df["quality"].values
print(y_df[:10])

# ### Data Preview

# In[55]:


plt.hist(y_df, range=(1, 10))

plt.xlabel('Ratings of wines')
plt.ylabel('Amount')
plt.title('Distribution of wine ratings')
plt.show()

# ### 1~10 Ratings to Binary Classification
# 
# Simplify the classification problem into a binary one: **good/bad**
# 
# Score < 7: bad(0); score >= 7: good(1).

# In[61]:


# sklearn can only deal with numpy arrys
y = np.array([1 if i>=7 else 0 for i in y_df])
print(y[:10])

# ### Random Forests Classifier Construction
# 
# Choose random forests method to do the classification.

# In[10]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

# ### Parameter Tuning
# 
# Parameter '**number of decision trees to construct**' is very important.
# 
# Iterate the 'number of decision trees'(**n_estimators**) parameter from 1 to 40.
# 
# For each iteration, use **cross_val_score** to compute its score 10 times with different splitting.

# In[11]:


scores = []

for num_trees in range(1,41):
    clf = RandomForestClassifier(n_estimators = num_trees)
    scores.append(cross_val_score(clf, X, y, cv=10))

# In[12]:


print(scores[0])
print(scores[1])

# In[13]:


sns.boxplot(data=scores)
plt.xlabel('Number of trees')
plt.ylabel('Classification score')
plt.title('Classification score as a function of the number of trees')
plt.show()

# ### Unbalanced Classification Evaluation

# In[14]:


good_ratio = sum(y) / len(y)
bad_ratio = 1 - good_ratio
print('Ratio of good wine: ', good_ratio)
print('Ratio of bad wine: ', bad_ratio)

# We can see that **the classes are unbalanced**.
# 
# Much more 'bad' wines
# 
# The **accuracy** metric may be misleading in this case, choose **F1** metric instead, which is less sensitive to imbalance. 
# 
# **F1: Harmonic mean of sensitivity & precision.**

# In[15]:


scores = []

for num_trees in range(1,41):
    clf = RandomForestClassifier(n_estimators = num_trees)
    scores.append(cross_val_score(clf, X, y, cv=10, scoring='f1'))

# In[16]:


sns.boxplot(data=scores)
plt.xlabel('Number of trees')
plt.ylabel('F1 score')
plt.title('F1 Scores as a function of the number of trees')
plt.show()

# The scores are clustered around the **40% mark**.
# 
# Set the number of decision trees to be 15. 

# In[142]:


clf = RandomForestClassifier(n_estimators = 15)
f1_score = cross_val_score(clf, X, y, cv=10, scoring='f1')
print(f1_score.mean())

# ### Probability Calibration
# 
# Compute the **predicted probabilities**.

# In[17]:


clf = RandomForestClassifier(n_estimators = 15)
clf.fit(X,y)

# clf.**predict()** gives the **predicted label**
# 
# clf.**predict_proba()** gives the **predicted probability**

# In[18]:


print(clf.predict(X)[:10])

# In[19]:


print(clf.predict_proba(X)[:10])

# Want to **predict label from predicted probability**.
# 
# Construct a prediction based on these predicted probabilities that labels all wines with a **predicted probability of being in class 1 > 0.5** with a 1 and 0 otherwise

# In[25]:


prediction = clf.predict(X)
prediction_from_proba = (clf.predict_proba(X)[:,1]>0.5).astype(int)

# **Compare**
# 
# the **constructed prediction** based on probabilities
# 
# with
# 
# the **classifier's prediction **

# In[26]:


correct = [1 if prediction_from_proba[i]== prediction[i] else 0 for i in range(len(prediction))]
print(sum(correct)/len(correct))

# In[27]:


prediction_from_proba = (clf.predict_proba(X)[:,1]>0.8).astype(int)
correct = [1 if prediction_from_proba[i]== prediction[i] else 0 for i in range(len(prediction))]
print(sum(correct)/len(correct))

# In[28]:


prediction_from_proba = (clf.predict_proba(X)[:,1]>0.1).astype(int)
correct = [1 if prediction_from_proba[i]== prediction[i] else 0 for i in range(len(prediction))]
print(sum(correct)/len(correct))

# ### Probability Calibration Function
# 
# Function to compute prediction from **trained classifier, training dataset** and **cutoff value(threshold)**.

# In[29]:


"""
cutoff_predict(clf, X, cutoff)

Inputs:
clf: a **trained** classifier object
X: a 2D numpy array of features
cutoff: a float giving the cutoff value used to convert
        predicted probabilities into a 0/1 prediction.

Output:
a numpy array of 0/1 predictions.
"""

def cutoff_predict(clf, X, cutoff):
    return (clf.predict_proba(X)[:,1] > cutoff).astype(int)

# ### Cutoff Value Tuning
# 
# **Evaluate different cutoff values under different train/test splittings** using **cross-validation**.

# In[41]:


def custom_f1(cutoff):
    def f1_cutoff(clf, X, y):
        ypred = cutoff_predict(clf, X, cutoff)
        return sklearn.metrics.f1_score(y, ypred)
        
    return f1_cutoff

# In[42]:


scores = []

for cutoff in np.arange(0.1,0.9,0.1):
    clf = RandomForestClassifier(n_estimators = 15)
    score_list = cross_val_score(clf, X, y, cv=10, scoring=custom_f1(cutoff))   
    scores.append(score_list)

# Using a **boxplot**, compare the **F1 scores** that correspond to each candidate **cutoff** value.
# 
# Choose cutoff value to be **0.2**. This is due to **class imbalance**, many fewer 'good' wine.

# In[54]:


sns.boxplot(x=np.arange(0.1,0.9,0.1), y=scores)
plt.xlabel('Cutoff values')
plt.ylabel('Custom F1 scores')
plt.title('Custom F1 scores as a function of the number of trees')
plt.show()

# ### Compare the Importance of Different Features
# 
# **Relative importance** of different features in the random forest.
# 
# Random forests allow us to compute a heuristic for determining how "important" a feature is in predicting a target. 
# 
# This heuristic measures the change in prediction accuracy if we take a given feature and permute (scramble) it across the datapoints in the training set. 
# 
# **The more the accuracy drops when the feature is permuted, the more "important" we can conclude the feature is.**

# In[66]:


clf = RandomForestClassifier(n_estimators=15)
clf.fit(X,y)

# Random forests provides **importance list**

# In[91]:


importance_list = clf.feature_importances_
print(importance_list)

# In[115]:


name_list = df.columns[:-1]
# Sort by the importance
importance_list, name_list = zip(*sorted(zip(importance_list, name_list)))
y_pos = np.arange(len(name_list))

print(name_list)
print(y_pos)

# In[116]:


plt.barh(y_pos,importance_list,align='center')
plt.yticks(range(len(name_list)),name_list)

plt.xlabel('Relative Importance in the Random Forest')
plt.ylabel('Features')
plt.title('Relative importance of Each Feature')
plt.show()

# ### Visualizing Classifiers Using Decision Surfaces
# 
# Decision surface visualizations are really only meaningful if they are plotted against inputs X that are **one- or two-dimensional(visualizable)**.
# 
# Choose the **2 most important features** (alcohol, sulphates) to visualize.
# 
# Subset the data matrix to include just the two features of highest importance.

# In[132]:


# Most important
important_list_1 = clf.feature_importances_
col1_index = list(important_list_1).index(max(important_list_1))
# Second most important
col2_index = list(important_list_1).index(sorted(clf.feature_importances_)[-2])

print(col1_index)
print(col2_index)

# In[134]:


X_imp = X[:,[col1_index, col2_index]]
print(X_imp)

# In[117]:


from sklearn.tree import DecisionTreeClassifier
import sklearn.linear_model
import sklearn.svm

def plot_decision_surface(clf, X_train, Y_train):
    plot_step=0.1
    
    if X_train.shape[1] != 2:
        raise ValueError("X_train should have exactly 2 columnns!")
    
    x_min, x_max = X_train[:, 0].min() - plot_step, X_train[:, 0].max() + plot_step
    y_min, y_max = X_train[:, 1].min() - plot_step, X_train[:, 1].max() + plot_step
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    clf.fit(X_train,Y_train)
    if hasattr(clf, 'predict_proba'):
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])    
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Reds)
    plt.scatter(X_train[:,0],X_train[:,1],c=Y_train,cmap=plt.cm.Paired)
    plt.show()

# In[137]:


classifiers = [DecisionTreeClassifier(),
               RandomForestClassifier(n_estimators=15),
               sklearn.svm.SVC(C=100.0, gamma=1.0)]

titleClassifer = ['Decision Tree Classifier', 'Random Forest Classifier', 
                  'Support Vector Machine']
for c in range(3):
    plt.title(titleClassifer[c])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plot_decision_surface(classifiers[c], X_imp, y)
