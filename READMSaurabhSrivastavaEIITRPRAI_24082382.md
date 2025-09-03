# IITRPRAI_24082382
IITRPRAI_24082382 AI Assignment 
AI for Market Trend Analysis
This project aims to build an AI system that analyzes historical financial data and predicts short-term market trends using machine learning techniques.

Project Objective
The objective is to predict market trends based on historical stock and financial indicators, specifically forecasting the next day's closing price direction (Up/Down).

Dataset and Features
The dataset used is historical stock data for Apple (AAPL), fetched using the yfinance library for the past 5 years. The raw data includes Open, High, Low, Close, Adjusted Close, and Volume.

Engineered Features:

We engineered a comprehensive set of technical indicators and lagged features:

Close price: The closing price of the stock.
SMA_20 (Simple Moving Average, 20 periods): Identifies general trend direction.
RSI_14 (Relative Strength Index, 14 periods): Momentum oscillator indicating overbought/oversold conditions.
MACD (Moving Average Convergence Divergence): Trend-following momentum indicator. Includes MACD line, Signal Line, and MACD Histogram.
EMA_20 (Exponential Moving Average, 20 periods): Smoother trend indicator giving more weight to recent prices.
ATR_14 (Average True Range, 14 periods): Volatility indicator.
Close_Lag1: Previous day's closing price.
Volume_Lag1: Previous day's trading volume.
(Note: Technical indicators were calculated manually due to compatibility issues with the pandas-ta library.)

Prediction Approach
The problem is framed as a binary classification task to predict the next day's price direction (Up/Down).

Data Splitting: A time-based split (80% train, 20% test) was used to maintain temporal order.
Class Imbalance Handling: Random Oversampling (ROS) was applied to the training data using imblearn to address class imbalance.
Model Exploration: We explored Logistic Regression (baseline), Random Forest, and XGBoost models.
Model Selection and Tuning: XGBoost, trained on enhanced features and resampled data, showed promising results and was selected for hyperparameter tuning using GridSearchCV with TimeSeriesSplit.
Evaluation and Insights
The models were evaluated using Accuracy, Confusion Matrix, and Classification Report, with a focus on Precision for Class 1 (Up Trend) and Recall for Class 0 (Down/Flat Trend) to minimize false trading signals.

Summary of Key Findings:

Random Oversampling successfully balanced the training data.
Enhanced feature engineering improved the performance of tree-based models (Random Forest and XGBoost), particularly in identifying downward/flat trends (higher Recall for Class 0).
XGBoost consistently outperformed Logistic Regression and Random Forest.
The Tuned XGBoost model achieved the best balance and overall performance:
Accuracy: ~0.5466
Precision (Class 1 - Up Trend): ~0.6020
Recall (Class 0 - Down/Flat Trend): ~0.6609
Despite improvements, the precision for predicting upward trends remains relatively low (~0.6020), meaning approximately 39.8% of the model's predictions for an upward trend are incorrect. This is a critical limitation for real-world trading.
Pitfalls Encountered:

pandas-ta import error (resolved by manual calculation).
Potential for Overfitting (mitigated by time-series split and regularization).
Market Noise (inherent challenge).
Class Imbalance (addressed by ROS, but potential issues remain).
Limitations and Future Scope
Current Limitations:

Relatively low precision for predicting upward trends.
Limited feature set (can be expanded).
Model complexity vs. interpretability trade-off.
Static model (requires retraining).
Future Scope:

Further hyperparameter tuning for the best model.
Explore advanced models (LSTMs, 1D CNNs, other ensembles).
Incorporate alternative data (news sentiment, economic indicators).
Experiment with different target variables (price return, multi-class trend).
Implement more robust backtesting strategies.
Focus on techniques specifically aimed at improving Class 1 precision.
How to Use This Notebook
Open the notebook in Google Colab.
Run the cells sequentially to load data, preprocess, engineer features, train and evaluate models, and generate the report.
Explore the code and outputs to understand the analysis steps and results.
