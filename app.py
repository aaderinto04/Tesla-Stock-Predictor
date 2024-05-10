import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

model = load_model('Nikky 1\OneDrive\Documents\StockPriceProj\StockPredictionsModel.keras.zip')


st.header('Tesla Stock Predictor')

df = pd.read_csv('TSLA.csv')

st.subheader('Stock Data')
st.write(df)

df_train = pd.DataFrame(df.Close[0: int(len(df)*0.80)])
df_test = pd.DataFrame(df.Close[int(len(df)*0.80): len(df)])

scaler = MinMaxScaler(feature_range=(0,1))

past_100days = df_train.tail(100)
df_test = pd.concat([past_100days, df_test])
df_test_s = scaler.fit_transform(df_test)

st.subheader('Price MA50')
moving_avg50 = df.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(moving_avg50, 'b')
plt.plot(df.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price MA50 vs MA 100')
moving_avg100 = df.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(moving_avg50, 'b')
plt.plot(moving_avg100, 'r')
plt.plot(df.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price MA100 vs MA 200')
moving_avg200 = df.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(moving_avg100, 'b')
plt.plot(moving_avg200, 'r')
plt.plot(df.Close, 'g')
plt.show()
st.pyplot(fig3)

x_test = []
y_test = []
for x in range(100, df_test_s.shape[0]):
    x_test.append(df_test_s[x-100:x])
    y_test.append(df_test_s[x,0])

x_test, y_test = np.array(x_test), np.array(y_test)


predicted_prices = model.predict(x_test)
#predicted_prices = scaler.inverse_transform(predicted_prices)
scale = 1/scaler.scale_
predicted_prices = predicted_prices*scale
y_test = y_test*scale

st.subheader('Actual Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.figure(figsize=(10,8))
plt.plot(predicted_prices, 'b', label = 'Predicted Prices')
plt.plot(y_test, 'r', label = 'Actual Price')
plt.title('Tesla Share Price')
plt.xlabel('Time')
plt.ylabel('Share Price')
plt.legend()
plt.show()    
st.pyplot(fig4)
