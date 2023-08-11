## Trading Predictions with LSTM + Volatility Indicators
A Streamlit web-app built to predict asset prices based on historical data.

### Architecture
The application processes a stock ticker and retrieves historical stock prices spanning the previous 365 days. It also incorporates extra functionalities, such as assessing the stock's volatility (beta), market volatility, and generating Bollinger Bands. The LSTM model is employed for training and produces two outputs: (1) the trained outcomes and (2) a price forecast for the upcoming 3 days. These outcomes are stored within an S3 bucket. The application then retrieves and compares the predicted performance with the actual prices.

On average, the model is able to achieve a ~94% accuracy rate with less than 6% price difference from the actual price and the predicted price for the desired assets.

<img width="505" alt="Screenshot 2023-08-10 at 19 25 30" src="https://github.com/yzse/px-app/assets/54381977/7eb6ad8b-df29-41e8-bf26-b437b01c76cb">

<img width="605" alt="Screenshot 2023-08-10 at 19 21 29" src="https://github.com/yzse/px-app/assets/54381977/2efc84a8-8875-4376-9f26-38bf8e8c4b70">
