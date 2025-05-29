from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))
    return scaled_data, scaler
