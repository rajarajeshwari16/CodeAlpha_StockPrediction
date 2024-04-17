#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import seaborn as sns


# In[3]:


df = pd.read_csv('GOOGL.csv')


# In[4]:


# Display the rows of the dataset


# In[5]:


print("Initial dataset:")
print(df.head())


# In[6]:


# Check for missing values


# In[7]:


print("\nChecking for missing values:")
print(df.isnull().sum())


# In[8]:


#Data Preprocessing


# In[9]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# In[10]:


#Histogram


# In[11]:


plt.figure(figsize=(20, 8))
plt.hist(df['Adj Close'], bins=20, color='LimeGreen')
plt.title('Histogram of Google Stock Prices')
plt.xlabel('Google Stocks')
plt.ylabel('Frequency')
plt.show()


# In[12]:


#TIme Series Trend


# In[13]:


plt.figure(figsize=(20, 8))
plt.plot(df['Adj Close'])
plt.title('Google Stocks Trend')
plt.show()


# In[14]:


#Feature Scaling: We scale the 'Close' prices using MinMaxScaler to normalize the data, which is a common preprocessing step for neural network-based models.


# In[15]:


closing_prices = df.filter(['Close']).values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(closing_prices)


# In[16]:


#Training Dataset: We split the dataset into training and testing sets, and create sequences of data with 60 time steps for training the LSTM model.


# In[17]:


train_data = scaled_prices[:int(len(scaled_prices) * 0.95), :]
train_features, train_labels = [], []

for i in range(60, len(train_data)):
    train_features.append(train_data[i-60:i, 0])
    train_labels.append(train_data[i, 0])

train_features, train_labels = np.array(train_features), np.array(train_labels)
train_features = np.reshape(train_features, (train_features.shape[0], train_features.shape[1], 1))


# In[18]:


# Building the LSTM Model: We construct a Sequential model with three LSTM layers and a Dense output layer.


# In[19]:


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_features.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))


# In[20]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[21]:


model.fit(train_features, train_labels, epochs=25, batch_size=32)


# In[22]:


# Prepare Testing Data: We prepare the testing data by creating sequences of data with 60 time steps for testing the LSTM model


# In[23]:


# Create the testing dataset


# In[24]:


test_data = scaled_prices[int(len(scaled_prices) * 0.95) - 60:, :]
x_test, y_test = [], closing_prices[int(len(closing_prices) * 0.95):, :]


# In[25]:


for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[26]:


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[27]:


training_set = df.iloc[:int(len(closing_prices) * 0.95)]
validation_set = df.iloc[int(len(closing_prices) * 0.95):]
validation_set.loc[:, 'Predictions'] = predictions.copy()


# In[28]:


# Visualizing Predictions


# In[29]:


# Visualize the predicted prices compared to actual prices


# In[30]:


plt.figure(figsize=(16, 8))
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(training_set['Close'], label='Training Data')
plt.plot(validation_set[['Close', 'Predictions']], label=['Actual Prices', 'Predicted Prices'])
plt.legend(loc='lower right')
plt.show()


# In[31]:


# Model Evaluation Metrics: We calculate the Mean Squared Error (MSE) as a metric to evaluate the performance of the LSTM model.


# In[32]:


mse_lstm = mean_squared_error(df['Close'][-len(predictions):], predictions)
print('\nLSTM Model Evaluation:')
print(f'MSE: {mse_lstm:.2f}')


# In[ ]:




