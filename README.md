## Trading Predictions with LSTM + Volatility Indicators
A Streamlit web-app built to predict asset prices based on historical data.

### Architecture
The application processes a stock ticker and retrieves historical stock prices spanning the previous 365 days. It also incorporates extra functionalities, such as assessing the stock's volatility (beta), market volatility, and generating Bollinger Bands. The LSTM model is employed for training and produces two outputs: (1) the trained outcomes and (2) a price forecast for the upcoming 3 days. These outcomes are stored within an S3 bucket, which then the application retrieves and compares the predicted performance with the actual prices.

On average, the model is able to achieve a ~94% accuracy rate with less than 6% price difference from the actual price and the predicted price for the desired assets.

<img width="490" alt="Screenshot 2023-12-06 at 00 14 03" src="https://github.com/yzse/px-app/assets/54381977/086ebee0-5471-4827-aef7-a3e98a459642">

<img width="503" alt="Screenshot 2023-12-06 at 00 14 12" src="https://github.com/yzse/px-app/assets/54381977/fcfc98a5-0908-45c9-b62d-39477f8a720f">
