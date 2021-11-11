#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns


# In[4]:


iris=sns.load_dataset('iris')


# In[5]:


iris.head()


# verideki tür değişkenini drop ile kaldırdık.

# In[8]:


X_iris= iris.drop('species', axis=1)
X_iris.shape


# hedef diziyi veri setinden çekelim 

# In[9]:


y_iris= iris['species']
y_iris.shape


# In[11]:


import matplotlib.pyplot as plt
import numpy as np


# In[12]:


rng=np.random.RandomState(42)


# In[13]:


x=10*rng.rand(50)
y=2*x-1+rng.randn(50)
plt.scatter(x,y)


# Grafikten 2 değişken arasında doğrusal bi ilişki olduğunu görüyoruz. Basit doğrusal regresyon yapıcaksak lineer sınıfını import etmeliyiz.

# In[14]:


from sklearn.linear_model import LinearRegression


# Model sınıfı belirledikten sonra bazı parametleri kendimiz belirleyebiliriz. Belirlediğimiz parametrelere hiper parametre denir. 

# hiper parametreler ön tanımlı parametrelerin üzerine yazılır yazmazsak ön parametreler geçerli olur. 

# In[17]:


model = LinearRegression (fit_intercept=True)
model


# X i düzenlememiz gerekiyor. 

# In[18]:


X=x[:,np.newaxis]
X.shape


# In[19]:


model.fit(X,y)


# Model kuruldu kullanılan parametreler ekrana yazıldı. Scikit learnde Modelin attribudelarına _ eklenir . 

# In[21]:


model.coef_


# In[22]:


model.intercept_


# Böylece basit regresyon değerler için sabit ve eğim değerlerini bulduk. 

# Modeli kurduktan sonra 2. adım yeni verileri değirlendirmektir. Bunun için predict metodu kullanılır. Önecelikle X eksenini bölelim ve X fit değişkenine atayalım. Sonra bir boyutlu dizi haline getirelim. Daha sonra bu değerleri modelin predict metodu ile tahmin edelim. 

# In[24]:


x_fit=np.linspace(-1,11)
X_fit=x_fit[:,np.newaxis]
y_fit= model.predict(X_fit)


# bulunan değerleri görselleştirelim. İlk önce x,y saçılım grafiğini yapalım. Sonra bu grafiğin üstüne x_fit ve y_fitin doğrusunu çizelim
# 

# In[26]:


plt.scatter(x,y)
plt.plot(x_fit,y_fit)


# Görüldüğü üzere veri setine göre regresyon doğrusu çizilmiş oldu. 

# Şimdi iris veri setinin eğitim setini kullanarak bir model oluşturmak. Daha sonra bu modeli kullanrak test verisindeki verilerin etiketini tahmin etmektir. 'Bayes yöntemi kullanılır' Bu yöntem hem hızlı hem de hiperparametre gerektirmiyor. 
# 

# İlk olarak  veri setini eğitim ve test olarak ikiye bölelim. Bu işlem için Scikit learn'ün bize sunduğu train_test_split ile yapılır.

# In[28]:


from sklearn.model_selection import train_test_split


# In[30]:


X_egitim,X_test, y_egitim, y_test = train_test_split(X_iris,y_iris, random_state= 1 )


# eğitim ve test için değişkenler oluştu. Şimdi model sınıfını çizelim.

# In[31]:


from sklearn.naive_bayes import GaussianNB


# In[37]:


model= GaussianNB()


# In[38]:


model.fit(X_egitim, y_egitim)


# Yeni veri için tahmin yapalım. Predict metodu çağrılır. 

# In[41]:


y_model = model.predict(X_test);


# Modelin doğruluk oranını bulalım. ( accuracy_score )

# In[47]:


from sklearn.metrics import accuracy_score


# In[48]:


accuracy_score(y_test, y_model)


# Bu modeli kullanarak %97 oranında doğru tahmin edebiliriz. 

# In[ ]:




