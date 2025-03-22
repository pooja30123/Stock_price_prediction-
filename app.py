#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# In[2]:


# Download daily stock data for the last 5 years
df = yf.download("TSLA", period="5y", interval="1d")

# Show first 5 rows
print(df.head())

# Check dataset size
print("Dataset shape:", df.shape)


# In[3]:


df.to_csv("stock_data_tesla.csv")
print("Dataset saved successfully!")


# In[4]:


plt.figure(figsize=(10, 5))
plt.plot(df['Close'], label="Close Price", color='green')
plt.title("Tesla Stock Price Over Time (Daily Data)")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.show()


# In[5]:


# Check for missing values
print(df.isnull().sum())

# Fill missing values using forward fill
df.fillna(method="ffill", inplace=True)


# In[6]:


scaler = MinMaxScaler(feature_range=(0, 1))

# Scale only the 'Close' price (since we're predicting it)
df_scaled = scaler.fit_transform(df[['Close']])

print("Scaled Data Sample:", df_scaled[:5])  # First 5 values after scaling


# In[7]:


train_size = int(len(df_scaled) * 0.8)
train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)


# In[8]:


def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])  # Last 60 days
        y.append(data[i + time_steps])  # Next day's price
    return np.array(X), np.array(y)

time_steps = 60  # Use last 60 days to predict next price
X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)


# In[9]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    
    LSTM(units=50),
    Dropout(0.2),
    
    Dense(units=1)  # Output layer (predict next price)
])

# Print model summary
model.summary()


# In[10]:


model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mse'])


# In[11]:


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32,validation_data=(X_test,y_test))


# In[12]:


# Predict stock prices on the test data
predicted_prices = model.predict(X_test)


# In[13]:


# Reshape predictions and test labels
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[14]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(actual_prices, color='green', label="Actual Stock Price")
plt.plot(predicted_prices, color='blue', linestyle='dashed', label="Predicted Stock Price")
plt.title("Stock Price Prediction using LSTM")
plt.xlabel("Time")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.show()


# In[15]:


# Get the last 60 days of data
last_60_days = df_scaled[-60:]  # Last 60 days (already scaled)

# Reshape to match LSTM input shape
X_future = np.array([last_60_days])
print("Future Input Shape:", X_future.shape)  # Should be (1, 60, 1)


# In[16]:


future_predictions = []

# Generate predictions for the next 7 days
for _ in range(7):
    predicted_price = model.predict(X_future)[0]  # Predict next price
    future_predictions.append(predicted_price)  # Store prediction

    # Add the predicted price to the sequence
    new_sequence = np.append(X_future[:, 1:, :], [[predicted_price]], axis=1)
    X_future = new_sequence.reshape(1, 60, 1)


# In[18]:


# Convert back to actual stock prices
future_predictions = scaler.inverse_transform(future_predictions)

print("Predicted Prices for the Next 7 Days:", future_predictions.flatten())


# In[19]:


# Plot the next 7 days of predicted prices
plt.figure(figsize=(10, 5))
plt.plot(future_predictions, marker='o', linestyle='dashed', color='red', label="Predicted Prices")
plt.title("Predicted Stock Prices for the Next 7 Days")
plt.xlabel("Days")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.show()


# In[20]:


import nbformat
from nbconvert import PythonExporter

def convert_ipynb_to_py(ipynb_path, py_path):
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    exporter = PythonExporter()
    python_script, _ = exporter.from_notebook_node(notebook)

    with open(py_path, "w", encoding="utf-8") as f:
        f.write(python_script)

# Example usage
convert_ipynb_to_py("StockPricePred.ipynb", "app.py")


# In[ ]:




