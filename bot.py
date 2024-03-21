import base64
import pandas as pd
import yfinance as yf
import joblib
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import streamlit as st
import plotly.graph_objects as go
#from openai import OpenAI

# Load the ARIMA model
arima_model = joblib.load('arima_model.joblib')

@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("IMAGES.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
    background-size: 180%;
    background-position: cover;
    background-repeat: no-repeat;
    background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Define functions for data fetching and prediction
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    latest_data = data.iloc[-1]
    return latest_data

def predict_price(ticker, start_date, end_date, steps=1):
    ticker = ticker.upper()
    latest_data = fetch_data(ticker, start_date, end_date)
    opening_price = latest_data['Open']
    high_price = latest_data['High']
    low_price = latest_data['Low']
    adj_closing_price = latest_data['Adj Close']
    vol = latest_data['Volume']

    user_data = {
        'Open': opening_price,
        'High': high_price,
        'Low': low_price,
        'Adj Close': adj_closing_price,
        'Volume': vol,
        'Year': latest_data.name.year,
        'Month': latest_data.name.month,
        'Day': latest_data.name.day
    }

    columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Year', 'Month', 'Day']
    input_data = pd.DataFrame([user_data], columns=columns)

    forecast = arima_model.get_forecast(steps=steps)  
    predicted_residuals = forecast.predicted_mean
    predicted_close = adj_closing_price + np.cumsum(predicted_residuals.values)

    # Generate date range for the predicted prices
    date_range = pd.date_range(start=end_date, periods=steps+1)
    predicted_prices = pd.DataFrame(predicted_close, index=date_range[1:], columns=['Predicted Close'])

    return predicted_prices

# Define the SMA strategy function
def sma_strategy(ticker, short_window, long_window):
    data = yf.download(ticker, period="1d", interval="1d")
    data['SMA_Short'] = data['Adj Close'].rolling(window=short_window).mean()
    data['SMA_Long'] = data['Adj Close'].rolling(window=long_window).mean()
    last_short_sma = data['SMA_Short'].iloc[-1]
    last_long_sma = data['SMA_Long'].iloc[-1]

    if last_short_sma > last_long_sma:
        return 'Buy'
    elif last_short_sma < last_long_sma:
        return 'Sell'
    else:
        return 'Hold'

# Streamlit UI with user inputs
def main():
    # Add a link to the style.css file
    with open("style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    st.title("Stock Price Prediction and Analysis App")

    ticker = st.selectbox("Select a ticker symbol:", ["BTC-USD", "ETH-USD", "LTC-USD"])

    date_range = st.date_input("Select a date range:", value=(pd.Timestamp('2022-01-01'), pd.Timestamp.today()))

    short_window = st.slider("Short SMA Window:", min_value=1, max_value=100, value=10)
    long_window = st.slider("Long SMA Window:", min_value=1, max_value=200, value=50)

    prediction_steps = st.number_input("Prediction Steps (ARIMA):", value=1, min_value=1)

    show_current_price = st.checkbox("Show Current Price")
    show_predicted_price = st.checkbox("Show Predicted Price")
    show_sma_analysis = st.checkbox("Show SMA Analysis")

    if st.button("Predict"):
        if ticker:
            # Pass start_date and end_date to predict_price
            predicted_closing_price = predict_price(ticker, date_range[0], date_range[1], steps=prediction_steps)
            if show_current_price:
                current_price = fetch_data(ticker, start_date=date_range[0], end_date=date_range[1])['Adj Close']
                st.write(f"Current Price for {ticker}:", current_price)

            # Plot historical and predicted prices
            data = yf.download(ticker, start=date_range[0], end=date_range[1])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Historical Prices'))
            fig.add_trace(go.Scatter(x=predicted_closing_price.index, y=predicted_closing_price['Predicted Close'], mode='lines', name='Predicted Prices'))
            fig.update_layout(title=f'Historical and Predicted Prices for {ticker}',
                              xaxis_title='Date',
                              yaxis_title='Price')
            st.plotly_chart(fig)

            if show_predicted_price:
                st.write(f"Predicted Closing Price for {ticker}:", predicted_closing_price)
            if show_sma_analysis:
                decision = sma_strategy(ticker, short_window, long_window)
                st.write(f"Trading Decision for {ticker}:", decision)
    st.title("Your Existing Streamlit App Title")

# Your existing Streamlit app code goes here...

# Chatbot integration
st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Execute the Streamlit app
if __name__ == '__main__':
    main()
