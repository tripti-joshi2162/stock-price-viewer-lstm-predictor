import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping  

st.title('📈 Stock Price Viewer + 15-Day LSTM Predictor')

companies = {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Google': 'GOOG',
    'Amazon': 'AMZN',
    'Tesla': 'TSLA',
    'Meta (Facebook)': 'META',
    'Netflix': 'NFLX'
}

company_name = st.selectbox('Select a company:', list(companies.keys()))
stock_symbol = companies[company_name]

start_date = pd.to_datetime(st.date_input('Start Date', value=pd.to_datetime('2020-01-01')))
end_date = pd.to_datetime(st.date_input('End Date', value=pd.to_datetime('2024-12-31')))

if end_date > pd.to_datetime("today"):
    st.warning("End date is in the future. Using today's date instead.")
    end_date = pd.to_datetime("today")

if st.button('Fetch Stock Data'):
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    if not data.empty:
        st.write(f"Showing stock data for *{company_name}* ({stock_symbol}) 📊")
        st.dataframe(data)
        st.line_chart(data['Close'])

        st.subheader("📉 Predicting next 15 days' Closing Prices using LSTM")

        df = data[['Close']].copy()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        sequence_length = 60
        X, y = [], []

        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i])

        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(X, y, epochs=500000, batch_size=32, verbose=1, callbacks=[early_stop])

        future_input = scaled_data[-sequence_length:]
        predictions = []

        for _ in range(15):
            pred_input = future_input.reshape((1, sequence_length, 1))
            pred = model.predict(pred_input, verbose=0)[0][0]
            predictions.append(pred)

            future_input = np.append(future_input[1:], pred).reshape(sequence_length)

        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        future_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=15)
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predicted_prices.flatten()})

        st.line_chart(forecast_df.set_index('Date'))
        st.dataframe(forecast_df)

    else:
        st.error('No data found. Please check the date range or try a different company.')



# .\venv\Scripts\activate