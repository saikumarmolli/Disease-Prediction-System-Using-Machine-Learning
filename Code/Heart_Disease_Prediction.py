#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('heart.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


data.describe()


# In[7]:


import seaborn as sns

corr = data.corr()

plt.figure(figsize = (15,15))
sns.heatmap(corr, annot = True)


# In[8]:


corr


# In[9]:


sns.set_style('whitegrid')
sns.countplot(x = 'target', data = data)


# In[10]:


# dataset = pd.get_dummies(data, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# In[11]:


dataset = data.copy()
dataset.head()


# In[12]:


X = dataset.drop(['target'], axis = 1)
y = dataset['target']


# In[13]:


X.columns


# In[14]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[15]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)


# In[16]:


pred = model.predict(X_test)
pred[:10]


# In[17]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)


# In[18]:


from sklearn.metrics import accuracy_score


# In[19]:


print(f"Accuracy of model is {round(accuracy_score(y_test, pred)*100, 2)}%")


# ## Hyperparameter Tuning

# In[20]:


from sklearn.model_selection import RandomizedSearchCV


# In[21]:


classifier = RandomForestClassifier(n_jobs = -1)


# In[22]:


from scipy.stats import randint
param_dist={'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200,300,400,500],
              'max_features':randint(1,31),
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               'min_samples_leaf':randint(1,31),
              }


# In[23]:


search_clfr = RandomizedSearchCV(classifier, param_distributions = param_dist, n_jobs=-1, n_iter = 40, cv = 9)


# In[24]:


search_clfr.fit(X_train, y_train)


# In[25]:


params = search_clfr.best_params_
score = search_clfr.best_score_
print(params)
print(score)


# In[26]:


claasifier=RandomForestClassifier(n_jobs=-1, n_estimators=400,bootstrap= False,criterion='gini',max_depth=5,max_features=3,min_samples_leaf= 7)


# In[27]:


classifier.fit(X_train, y_train)


# In[28]:


confusion_matrix(y_test, classifier.predict(X_test))


# In[29]:


print(f"Accuracy is {round(accuracy_score(y_test, classifier.predict(X_test))*100,2)}%")


# In[30]:


import pickle
pickle.dump(classifier, open('heart.pkl', 'wb'))

