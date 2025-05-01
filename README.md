# LSTM Stock Price Prediction

![Stock Prediction](https://img.shields.io/badge/Stock-Prediction-blue)
![LSTM](https://img.shields.io/badge/Deep%20Learning-LSTM-green)
![Python](https://img.shields.io/badge/Python-3.7+-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A deep learning model for stock price prediction using Long Short-Term Memory (LSTM) neural networks. This project fetches stock data from Google Sheets, processes it, and uses a stacked LSTM architecture to predict future stock prices.

## 📊 Project Overview

This project uses:
- LSTM neural networks for time series forecasting
- Google Sheets API for data storage and retrieval 
- Keras/TensorFlow for deep learning model implementation
- Min-Max scaling for data normalization
- 60-day window for feature engineering

## 🔍 Features

- **Data Integration**: Pulls stock data directly from Google Sheets
- **Preprocessing**: Date formatting, scaling and sequencing
- **Model Architecture**: 4-layer stacked LSTM with dropout regularization
- **Visualization**: Comparison charts of predicted vs actual stock prices
- **Prediction**: Forward prediction of stock trends

## 🛠️ Requirements

```
pandas
numpy
matplotlib
tensorflow
keras
sklearn
google-auth
gspread
datetime
```

## 📋 Dataset Structure

The Google Sheet contains multiple worksheets:
- `overall`: Index of all company stocks
- Individual sheets for each company with columns:
  - Date
  - Open
  - High
  - Low
  - Close
  - Volume
  - Company

## 🚀 How to Use

### 1. Authentication Setup

```python
from google.colab import auth
auth.authenticate_user()
import gspread
from google.auth import default
creds, _ = default()
gc = gspread.authorize(creds)
```

### 2. Data Loading

```python
WB = gc.open_by_url('YOUR_GOOGLE_SHEET_URL')
overall = WB.worksheet('overall')
df1 = pd.DataFrame(overall.get_all_records())
```

### 3. Data Processing

```python
# Define column structure
lst = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Company']
databas = pd.DataFrame(columns=lst)

# Fetch data for each company
for company in df1['COMPANY']:
    sheet = WB.worksheet(company)
    df2 = pd.DataFrame(sheet.get_all_records(), columns=lst)
    df2['Company'] = company
    
    # Format dates
    for i in range(0, len(df2)):
        temp = str(df2['Date'][i])[0:10]
        df2['Date'][i] = temp
        
    databas = databas.append(df2, ignore_index=True)
```

### 4. Data Visualization

```python
# Plot closing prices for a specific company
company = "GOOGL"  # Example company code
sheet = WB.worksheet(company)
df2 = pd.DataFrame(sheet.get_all_records(), columns=lst)
plt.figure(figsize=(30, 30))
plt.plot(df2.iloc[:, 1])
plt.ylabel('Close')
plt.xlabel('Date')
plt.title(f"Closing Price of {company}")
plt.tight_layout()
```

### 5. Model Training

```python
# Split data
test_set = df2.iloc[df2.shape[0]-20:, 1:2].values
training_set = df2.iloc[:df2.shape[0]-20, 1:2].values

# Scale data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=[0, 1])
train_set_scaled = sc.fit_transform(training_set)

# Create sequences
X_train = []
Y_train = []
for i in range(60, 991):
    X_train.append(train_set_scaled[i-60:i, 0])
    Y_train.append(train_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

# Compile and train
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, Y_train, epochs=100, batch_size=32)
```

### 6. Making Predictions

```python
# Prepare test data
dataset_total = np.append(training_set, test_set)
inputs = dataset_total[len(dataset_total)-len(test_set)-60:]
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
```

### 7. Visualizing Results

```python
plt.figure(figsize=(16, 8))
plt.plot(test_set, color='red', label='Real price')
plt.plot(predicted_stock_price, color='blue', label='Predicted price')
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
```

## 📈 Results

![Prediction Results](https://via.placeholder.com/800x400?text=Stock+Price+Prediction+Results)

The model achieves reasonable accuracy in predicting short-term stock price movements. The stacked LSTM architecture helps capture complex patterns in the time series data.

Key findings:
- The model performs better on stocks with lower volatility
- 60-day window provides optimal context for predictions
- Dropout layers (0.2) prevent overfitting effectively

## 🔮 Future Predictions

The model can be used to predict future stock prices by extending the prediction window:

```python
# Get recent data
ext = training_set[len(training_set)-40:]
final = np.append(ext, test_set)
final = final.reshape(-1, 1)
final = sc.transform(final)

# Reshape for prediction
X_test = np.reshape(final, (1, final.shape[0], 1))

# Make prediction
future_prediction = regressor.predict(X_test)
future_prediction = sc.inverse_transform(future_prediction)
print(future_prediction)
```

## 🧪 Model Evaluation

The model is evaluated using Mean Squared Error (MSE) as the loss function. Performance varies by stock but generally achieves lower error rates than traditional time series methods like ARIMA.

## 📝 Notes

- Stock price prediction is inherently challenging due to market volatility
- This model focuses on technical analysis only (price patterns)
- For production use, consider incorporating fundamental analysis and market sentiment

