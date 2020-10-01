#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train_X = np.genfromtxt(r'C:\Users\Pratik\Desktop\ML\Projects\linear_regression\train_X_lr.csv', delimiter=',', dtype=np.float64, skip_header=1)
train_Y = np.genfromtxt(r'C:\Users\Pratik\Desktop\ML\Projects\linear_regression\train_Y_lr.csv', delimiter=',', dtype=np.float64)


# In[4]:


train_X.shape


# In[5]:


train_X[:3]


# In[6]:


train_Y.shape


# In[7]:


train_Y[:3]


# In[8]:


def import_data() :
    X = np.genfromtxt(r'C:\Users\Pratik\Desktop\ML\Projects\linear_regression\train_X_lr.csv', delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt(r'C:\Users\Pratik\Desktop\ML\Projects\linear_regression\train_Y_lr.csv', delimiter=',', dtype=np.float64)
    return X, Y


# In[ ]:





# In[9]:


def compute_gradient_of_cost_function(X, Y, W) :
    Y_pred = np.dot(X,W)
    difference = Y_pred - Y
    dW = (1/len(X))*(np.dot(X.T, difference))
    dw = dW
    return dW


# In[10]:


def compute_cost(X, Y, W) :
    Y_pred = np.dot(X,W)
    difference = Y_pred - Y
    cost = (1/(2*len(X)))*(np.dot(difference.T,difference))
    return cost[0][0]


# In[11]:



def optimize_weights_using_gradient_descent(X, Y, W, num_iterations, learning_rate) :
    
    for i in range(num_iterations) :
        dW = compute_gradient_of_cost_function(X, Y, W)
        W = W - (learning_rate * dW)
        cost = compute_cost(X, Y, W)
        print(i, cost)
    return W


# In[12]:


def train_model(X, Y) :
    X = np.insert(X, 0, 1, axis = 1)
    Y = Y.reshape(len(X), 1)
    W = np.zeros((X.shape[1], 1))
    W = optimize_weights_using_gradient_descent(X, Y, W, 100, 0.00021)
    return W


# In[23]:



    
def optimize_weights_using_gradient_descent(X, Y, W, num_iterations, learning_rate) :
    previous_iter_cost = 0
    iter_no = 0
    while True :
        iter_no += 1
        dW = compute_gradient_of_cost_function(X, Y, W)
        W = W - (learning_rate * dW)
        cost = compute_cost(X, Y, W)
          

        if abs(previous_iter_cost - cost) < 0.000001 :
            print(iter_no, cost)
            break
            
        if iter_no % 10000 == 0 :
            print(iter_no, cost)        
            
        
        previous_iter_cost = cost
    return W


# In[24]:


def save_model(weights,weights_file_name):
    with open(weights_file_name,'w') as weights_file:
        wr=csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()

if __name__== "__main__":
    X,Y=import_data()
    weights=train_model(X,Y)
    
    save_model(weights,r"C:\Users\Pratik\Desktop\ML\Projects\linear_regression\WEIGHTS_FILE.csv")


# In[ ]:




