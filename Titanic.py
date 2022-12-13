#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv(r'C:\Users\sudhirb\Downloads\train.csv')


# In[3]:


train.head()


# In[4]:


## missing values


# In[5]:


train.isnull()


# In[6]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[7]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[8]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[9]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[10]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[11]:


train['Age'].hist(bins=30,color='darkred',alpha=0.3)


# In[12]:


sns.countplot(x='SibSp',data=train)


# In[13]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[14]:


## data cleaning


# In[15]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[16]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[17]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[18]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[19]:


train.drop('Cabin',axis=1,inplace=True)


# In[20]:


train.head()


# In[21]:


train.dropna(inplace=True)


# In[22]:


## converting


# In[23]:


train.info()


# In[24]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[25]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[26]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[27]:


train.head()


# In[28]:


train = pd.concat([train,sex,embark],axis=1)
train.head()


# In[29]:


## model


# In[30]:


##splitting data


# In[31]:


train.drop('Survived',axis=1).head()


# In[32]:


train['Survived'].head()


# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[34]:


##prediction


# In[35]:


from sklearn.linear_model import LogisticRegression


# In[36]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[37]:


predictions = logmodel.predict(X_test)


# In[38]:


from sklearn.metrics import confusion_matrix


# In[39]:


accuracy=confusion_matrix(y_test,predictions)
accuracy


# In[40]:


from sklearn.metrics import accuracy_score


# In[41]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# In[42]:


predictions


# In[43]:


from sklearn.metrics import classification_report


# In[44]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[ ]:




