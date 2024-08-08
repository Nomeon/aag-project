import pandas as pd
from datetime import timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA

# assumes complete preprocessing-pipeline with added features and completed clustering


# helper-function for creating the data input in LSTM format
# takes the input data and creates a descriptive feature including t past observations to derive the t+1th value
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# predict the net revenue per cluster for the upcoming weeks (6 months)
def predictRevenuePerCluster(clustered_customers_df: pd.DataFrame, allOrders: pd.DataFrame, predictionType: str = "LSTM", clusterID: int = 1, modeltype: int = 1) -> pd.DataFrame:
    
    """Takes the complete dataset and the assigned cluster for each customer and creates predicts the future net revenue values on a weekly basis per cluster for the next 6 months.

    Args:
      clustered_customers_df (pd.DataFrame): The output dataframe from the clusterRFM()-function including CustomerID and assigned cluster.
      allOrders (pd.DataFrame): The preprocessed DataFrame. Necessary Columns are "OrderDate", "CustomerID", "NetRevenue"..
      predictionType (str): Choice of prediction model approach. Either "LSTM" or "ARIMA".
      clusterID (int): Choice of cluster to predict for, that is the number of the respective cluster in the clustered_customers_df.
      modeltype (int): Choice of specific NN architecture design for the LSTM prediction. Either 1 for a sequential model including 1 LSTM layer and the Huber-loss-function, 2 for a sequential model including 2 LSTM layers, 2 Dropout layers, and the MSE-loss-function.

    Returns:
      pd.DataFrame: A DataFrame containing the predicted future revenue values. For the LSTM model this includes the original data, the training, the testing, and the predicted data. For the ARIMA model this includes the original data, the fitting data, and the predicted data.
    """
    
    
    ### prepare the given datasets
    clustered_customers_df.CustomerID = clustered_customers_df.CustomerID.astype(str)
    c_subsets = {}
    for num in clustered_customers_df['cluster'].unique():
        c_subsets[num] = clustered_customers_df[clustered_customers_df['cluster'] == num]
    allOrders["CustomerID"] = allOrders["CustomerID"].astype(str)
    allOrders["OrderDate"] = pd.to_datetime(allOrders['OrderDate'], format='%Y-%m-%d')
    
    # get relevant data for specified cluster from the original order dataset
    orders_subsets = allOrders[allOrders['CustomerID'].isin(c_subsets[(clusterID-1)]['CustomerID'])]
    orders_subsets["OrderDate"] = pd.to_datetime(orders_subsets['OrderDate'], format='%Y-%m-%d')
        
    # prepare the time series data, aggregate revenue per day
    customlevelpred = orders_subsets.sort_values(by="OrderDate")
    customlevelpred = customlevelpred.groupby('OrderDate')['NetRevenue'].sum().reset_index()
    # aggregate on a weekly basis
    customlevelpred = customlevelpred.groupby([pd.Grouper(key='OrderDate', freq='W-MON', label='left')])['NetRevenue'].sum().reset_index().sort_values('OrderDate')

    # drop missing values in one of the sets
    customlevelpred = customlevelpred[:130]

    
    ### choose type of model for predictions as defined in the function call and predict futre revenue
    
    # option 1: LSTM
    if (predictionType == "LSTM"):
        ### LSTM data preparation based on https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

        # get input data
        lstmdatainput = customlevelpred
        tf.random.set_seed(5)

        # transform integer into float for NN
        dataframe = lstmdatainput["NetRevenue"]
        dataset = dataframe.values
        dataset = dataset.astype('float32')

        # normalize the dataset using a MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset.reshape(-1, 1))

        # create training and testing split (we assume that we have 2 years of training and half a year for testing)
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

        # define the lag, i.e., the number of previous weeks to include in the creation of the current observation
        # as the autocorrelation analysis implied, there is a slight autocorrelation within the first few lags
        # thus some more are used here
        look_back = 10
        # create actual datasets
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        
        # model option with best parameters as proposed from grid search
        if (modeltype == 1):
            # create and fit the LSTM network
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.LSTM(24, activation='relu', input_shape=(1, look_back)))
            model.add(tf.keras.layers.Dense(1))
            model.compile(loss='huber', optimizer='adam', metrics=['mse', 'mae', 'mape'])
            model.fit(trainX, trainY, epochs=250, batch_size=4, verbose=2)
            
        # model option to include dropouts for handling overfitting
        elif (modeltype == 2):
            # create and fit the LSTM network
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.LSTM(100, activation='relu', input_shape=(1, look_back), return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.LSTM(50, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(1))
            model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])
            model.fit(trainX, trainY, epochs=250, batch_size=4, verbose=2)
        
        # predict on given data
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])
        
        
        ### predict future values (after training and testing, i.e.,, the future values, we want to know)
        
        # prepare the last sequence from the dataset
        last_sequence = dataset[-look_back:]
        # list to store the predictions
        future_predictions = []

        # predict the next 26 weeks (half a year)
        for _ in range(26):
            # reshape the last sequence to the shape [samples, time steps, features]
            input_seq = np.reshape(last_sequence, (1, 1, look_back))
            # predict the next value
            next_value = model.predict(input_seq)
            # append the predicted value to the predictions list
            future_predictions.append(next_value[0, 0])
            # update the last sequence by removing the first value and appending the predicted value
            last_sequence = np.append(last_sequence[1:], next_value)
            # reshape back to (look_back,) for the next iteration
            last_sequence = last_sequence.reshape((look_back,))

        # convert predictions
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions_original_scale = scaler.inverse_transform(future_predictions)
        
        # create date indices for the new predictions in the final dataset
        dates = lstmdatainput["OrderDate"]
        last_date = dates.iloc[-1]
        future_dates = [last_date + timedelta(weeks=i) for i in range(1, 27)]
        future_dates = pd.to_datetime(future_dates, format='%Y-%m-%d')
        # combine original dates with future dates
        extended_dates = pd.Series(np.concatenate([dates.values, future_dates]))
        extended_dates = pd.to_datetime(extended_dates, format='%Y-%m-%d')

        # array to hold the dataset including the future predictions
        extended_dataset = np.append(dataset, future_predictions, axis=0)
        # inverse transform the extended dataset
        extended_dataset_original_scale = scaler.inverse_transform(extended_dataset)

        # array to hold the future predictions
        futurePredictPlot = np.empty_like(extended_dataset)
        futurePredictPlot[:, :] = np.nan
        futurePredictPlot[len(dataset):, :] = future_predictions_original_scale
        # shift train predictions
        trainPredictPlot = np.empty_like(extended_dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
        # shift test predictions
        testPredictPlot = np.empty_like(extended_dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

        # combine extended dataset with dates for final output
        extended_dataset_with_dates = pd.DataFrame({
            'Date': extended_dates,
            'Revenue': extended_dataset_original_scale.flatten(),
            'Train' : trainPredictPlot.flatten(),
            'Test' : testPredictPlot.flatten(),
            'Predict' : futurePredictPlot.flatten(),
        })
     
        # final prediction output
        return extended_dataset_with_dates
    
    # option 2: ARIMA
    elif (predictionType == "ARIMA"):
        # set dates as indices to create time series
        customlevelpred.set_index('OrderDate', inplace=True)

        # fit the ARIMA model including specifications to consider the autocorrelation/lag, the season and stationarity
        model = ARIMA(customlevelpred["NetRevenue"], order=(5, 2, 5), seasonal_order=(0, 0, 0, 52), enforce_stationarity = True)
        model_fit = model.fit()
        
        ### predict future values (after training and testing, i.e.,, the future values, we want to know)
        
        # forecast the next 26 weeks (half a year)
        forecast_steps = 26
        forecast = model_fit.forecast(steps=forecast_steps)
        # DataFrame to hold the forecasted values
        last_date = customlevelpred.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_steps, freq='W-MON')[1:]
        forecast_df = pd.DataFrame(forecast, index=forecast_dates)

        # combine the original data with the forecast for the final output
        customlevelpred["ARIMA"] = model_fit.fittedvalues
        combined_df = pd.concat([customlevelpred, forecast_df], axis=0)
        
        # final prediction output
        return combined_df
    
# to be run in main():

# load relevant data if not given through previously run functions
#clustered_customers_df = pd.read_csv('clusterAssignments.csv', sep = ",")
#allOrders = pd.read_csv("top25OrdersRevenue.csv")

# get predictions:
# predictions = predictRevenuePerCluster(clustered_customers_df, allOrders, "LSTM", 2,2)

