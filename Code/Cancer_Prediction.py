#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('cancer.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


import seaborn as sns


# In[6]:


sns.set_style('whitegrid')
sns.countplot(x = 'diagnosis', data = data)


# In[7]:


dataset = data
dataset['diagnosis'].replace(['M','B'], [1,0], inplace = True)


# In[8]:


dataset.drop('Unnamed: 32',axis = 1, inplace = True)


# In[9]:


corr = dataset.corr()
plt.figure(figsize = (25,25))
sns.heatmap(corr, annot = True)


# In[10]:


dataset.corr()


# In[11]:


dataset.drop(['id','symmetry_se','smoothness_se','texture_se','fractal_dimension_mean'], axis = 1, inplace = True)


# In[12]:


dataset.head()


# In[13]:


plt.figure(figsize = (25,25))
sns.heatmap(dataset.corr(), annot = True)


# In[14]:


X = dataset.drop('diagnosis', axis = 1)
y = dataset['diagnosis']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[17]:


print("Train Set: ", X_train.shape, y_train.shape)
print("Test Set: ", X_test.shape, y_test.shape)


# In[18]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)


# In[19]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[20]:


confusion_matrix(y_test, model.predict(X_test))


# In[21]:


print(f"Accuracy is {round(accuracy_score(y_test, model.predict(X_test))*100,2)}")


# ## Applying Hyperparameter Tuning

# In[22]:


from sklearn.model_selection import RandomizedSearchCV


# In[23]:


classifier = RandomForestClassifier(n_jobs = -1)


# In[24]:


from scipy.stats import randint
param_dist={'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200,300,400,500],
              'max_features':randint(1,27),
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               'min_samples_leaf':randint(1,27),
              }


# In[25]:


search_clfr = RandomizedSearchCV(classifier, param_distributions = param_dist, n_jobs=-1, n_iter = 40, cv = 9)


# In[26]:


search_clfr.fit(X_train, y_train)


# In[27]:


params = search_clfr.best_params_
score = search_clfr.best_score_
print(params)
print(score)


# In[28]:


claasifier=RandomForestClassifier(n_jobs=-1, n_estimators=200,bootstrap= True,criterion='gini',max_depth=20,max_features=8,min_samples_leaf= 1)


# In[29]:


classifier.fit(X_train, y_train)


# In[30]:


confusion_matrix(y_test, classifier.predict(X_test))


# In[31]:


print(f"Accuracy is {round(accuracy_score(y_test, classifier.predict(X_test))*100,2)}%")


# In[32]:


import pickle
pickle.dump(classifier, open('breast_cancer.pkl', 'wb'))

