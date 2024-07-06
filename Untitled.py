#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


df=pd.read_csv("titanic.csv")


# In[13]:


df


# In[14]:


df.head()


# In[58]:


df.head(15)


# In[16]:


df.shape


# In[17]:


df.describe()


# In[21]:


df['Survived'].value_counts()


# In[25]:


sns.countplot(x=df['Survived'], hue=df['Pclass'])


# In[23]:


df["Sex"]


# In[26]:


sns.countplot(x=df['Sex'], hue=df['Survived'])


# In[27]:


df.groupby('Sex')[['Survived']].mean()


# In[28]:


df['Sex'].unique()


# In[29]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['Sex']= labelencoder.fit_transform(df['Sex'])

df.head()


# In[30]:



df['Sex'], df['Survived']


# In[31]:


sns.countplot(x=df['Sex'], hue=df["Survived"])


# In[32]:


df.isna().sum()


# In[33]:


df=df.drop(['Age'], axis=1)


# In[34]:


df_final = df
df_final.head(10)


# In[35]:


X= df[['Pclass', 'Sex']]
Y=df['Survived']


# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[37]:


from sklearn.linear_model import LogisticRegression

log = LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)


# In[38]:


pred = print(log.predict(X_test))


# In[39]:


print(Y_test)


# In[57]:


import warnings
warnings.filterwarnings("ignore")

res= log.predict([[1,0]])

if(res==0):
  print("So Sorry! Not Survived")
else:
  print("Survived")


# In[ ]:





# In[ ]:




