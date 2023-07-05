#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[10]:


X = [[22,20000],[25,30000],[35,40000],[28,60000],[45,80000],[52,90000],[48,60000],[55,70000]]
y = [0,0,0,0,1,1,1,1]

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=5)

classifier = GaussianNB()

classifier.fit(X_train,Y_train)

y_pred = classifier.predict(X_test)

print(X_train)

accuracy = accuracy_score(Y_test,y_pred)

print("Accuracy:",accuracy)


# In[ ]:




