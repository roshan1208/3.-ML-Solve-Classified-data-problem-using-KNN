#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('KNN_Project_Data',index_col=0)


# In[5]:


df.head()


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


scaler = StandardScaler()


# In[8]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[9]:


df_feat = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[10]:


df_feat


# In[11]:


df_feature = pd.DataFrame(df_feat,columns=df.columns[:-1])


# In[12]:


df_feature.head()


# In[13]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(df_feature, df['TARGET CLASS'], test_size=0.3, random_state=101)


# In[18]:


X_train


# In[19]:


X_test


# In[44]:


y_train


# In[45]:


y_test


# In[55]:


from sklearn.neighbors import KNeighborsClassifier


# # Let's first find range of K for which error_rate will minimum

# In[48]:


error_rate =[]
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    predd = knn.predict(X_test)
    error_rate.append(np.mean(predd!=y_test))
    
    


# In[56]:


sns.set_palette('GnBu_r')
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',ls='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error rate vs K')
plt.xlabel('K')
plt.ylabel('Error rate')


# # After evaluting range of K we got to know that at k=37 we are getting least error_rate so.........
#    

# In[50]:


knn = KNeighborsClassifier(n_neighbors=37)


# In[51]:


knn.fit(X_train,y_train)


# In[52]:


predd = knn.predict(X_test)


# In[53]:


from sklearn.metrics import confusion_matrix, classification_report


# In[54]:


print('Confusion report')
print(confusion_matrix(y_test,predd))
print('\n')
print('classfication report')
print(classification_report(y_test,predd))


# In[ ]:




