import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


# Helper-function for creating the data input in LSTM format
# Takes the input data and creates a descriptive feature including t past observations to derive the t+1th value
def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)



def predict_quantity_per_cluster(rfc_clustering: pd.DataFrame, itemDataset: pd.DataFrame,
                                 prediction_type: str = "LSTM", 
                                 distribution_center: str = None, product_subgroup: str = None,
                                 pred_weeks: int = 52) -> pd.DataFrame:
    """
    Predicts the quantity per cluster based on the provided distribution center and optionally, the product subgroup.

    Args:
        rfc_clustering (pd.DataFrame): The DataFrame containing the clustered items.
        itemDataset (pd.DataFrame): The preprocessed DataFrame.
        prediction_type (str): The type of prediction model to be used, default is "LSTM".
        distribution_center (str): The distribution center for which the prediction is made (must be provided).
        product_subgroup (str or None): The specific product subgroup for prediction, if provided.
        pred_weeks (int): The number of weeks for which the prediction is made.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted quantities.
    """
    
    # Ensure that DistributionCenter is provided
    if distribution_center is None:
        raise ValueError("DistributionCenter must be provided.")
    
    clustered_items_df = rfc_clustering.copy()
    clustered_items_df['ProductSubgroup'] = clustered_items_df['ProductSubgroup'].astype(str)

    # Create subsets based on DistributionCenter
    c_subsets = {}
    for num in clustered_items_df['DistributionCenter'].unique():
        c_subsets[num] = clustered_items_df[clustered_items_df['DistributionCenter'] == num]

    # Preparing dataset for specific DistributionCenter predictions 
    if not product_subgroup:  # Correct condition to check if ProductSubgroup is not provided
        items_subsets = itemDataset.merge(
            c_subsets[distribution_center][['ProductSubgroup', 'DistributionCenter']], 
            on=['ProductSubgroup', 'DistributionCenter']
        )
    
    # Preparing dataset for specific ProductSubgroup in specific DistributionCenter predictions
    else:
        items_subsets = itemDataset.merge(
            c_subsets[distribution_center][c_subsets[distribution_center]['ProductSubgroup'] == product_subgroup][['ProductSubgroup', 'DistributionCenter']],
            on=['ProductSubgroup', 'DistributionCenter']
        )
    
    # Convert OrderDate to datetime
    items_subsets['OrderDate'] = pd.to_datetime(items_subsets['OrderDate'], format='%Y-%m-%d')
    
    # Sort the dataset by OrderDate
    itemlevelpred = items_subsets.sort_values(by='OrderDate')
    
    # Aggregate by week and sum up the NetRevenue and Quantity
    itemlevelpred = itemlevelpred.groupby(['OrderDate'])[['NetRevenue', 'Quantity']].sum().reset_index()
    itemlevelpred = itemlevelpred.groupby([pd.Grouper(key='OrderDate', freq='W-MON', label='left')])[['NetRevenue', 'Quantity']].sum().reset_index().sort_values('OrderDate')


    # Option 1: LSTM
    if (prediction_type == "LSTM"):

        # Get input data
        lstmdatainput = itemlevelpred.drop('NetRevenue', axis = 1)
        tf.random.set_seed(5)
                
        # Transform integer into float for NN
        dataframe = lstmdatainput['Quantity']
        dataset = dataframe.values
        dataset = dataset.astype('float32')

        # Normalize the dataset using a MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset.reshape(-1, 1))

        # Create training and testing split (we assume that we have 2 years of training and half a year for testing)
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

        # Define the lag, i.e., the number of previous weeks to include in the creation of the current observation
        # as the autocorrelation analysis implied, there is a slight autocorrelation within the first few lags
        # thus some more are used here
        look_back = 10
        # Create actual datasets
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        # Reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # Model option with best parameters as proposed from grid search
        # Create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(1, look_back), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(optimizer='Adam', loss='mse', metrics=['mse', 'mae', 'mape'])
        model.fit(trainX, trainY, epochs=150, batch_size=1, verbose=2)
        
        # Predict on given data
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        # Invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])
        
        # Prepare the last sequence from the dataset
        last_sequence = dataset[-look_back:]
        # List to store the predictions
        future_predictions = []

        # Predict the next n weeks
        for _ in range(pred_weeks):
            # Reshape the last sequence to the shape [samples, time steps, features]
            input_seq = np.reshape(last_sequence, (1, 1, look_back))
            # Predict the next value
            next_value = model.predict(input_seq)
            # Append the predicted value to the predictions list
            future_predictions.append(next_value[0, 0])
            # Update the last sequence by removing the first value and appending the predicted value
            last_sequence = np.append(last_sequence[1:], next_value)
            # Reshape back to (look_back,) for the next iteration
            last_sequence = last_sequence.reshape((look_back,))

        # Convert predictions
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions_original_scale = scaler.inverse_transform(future_predictions)
        
        # Create date indices for the new predictions in the final dataset
        dates = lstmdatainput['OrderDate']
        last_date = dates.iloc[-1]
        future_dates = [last_date + timedelta(weeks=i) for i in range(1, (pred_weeks+1))]
        future_dates = pd.to_datetime(future_dates, format='%Y-%m-%d')
        # Combine original dates with future dates
        extended_dates = pd.Series(np.concatenate([dates.values, future_dates]))
        extended_dates = pd.to_datetime(extended_dates, format='%Y-%m-%d')

        # Array to hold the dataset including the future predictions
        extended_dataset = np.append(dataset, future_predictions, axis=0)
        # Inverse transform the extended dataset
        extended_dataset_original_scale = scaler.inverse_transform(extended_dataset)

        # Array to hold the future predictions
        futurePredictPlot = np.empty_like(extended_dataset)
        futurePredictPlot[:, :] = np.nan
        futurePredictPlot[len(dataset):, :] = future_predictions_original_scale
        # Shift train predictions
        trainPredictPlot = np.empty_like(extended_dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
        # Shift test predictions
        testPredictPlot = np.empty_like(extended_dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

        # Combine extended dataset with dates for final output
        extended_dataset_with_dates = pd.DataFrame({
            'Date': extended_dates,
            'Quantity': extended_dataset_original_scale.flatten(),
            'Train' : trainPredictPlot.flatten(),
            'Test' : testPredictPlot.flatten(),
            'Predict' : futurePredictPlot.flatten(),
        })
    
        # Final prediction output
        return extended_dataset_with_dates
        
        
        
    # Option 2: ARIMA
    elif (prediction_type == 'ARIMA'):
    # Set dates as indices to create time series
        itemlevelpred.set_index('OrderDate', inplace=True)

        # Fit the ARIMA model including specifications to consider the autocorrelation/lag, the season and stationarity
        model = ARIMA(itemlevelpred['Quantity'], order=(5, 2, 5), seasonal_order=(0, 0, 0, 52), enforce_stationarity = True)
        model_fit = model.fit()
        
        # Predict future values (after training and testing, i.e.,, the future values, we want to know)
        
        # Forecast the next n weeks
        forecast_steps = pred_weeks
        forecast = model_fit.forecast(steps=forecast_steps)
        # DataFrame to hold the forecasted values
        last_date = itemlevelpred.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_steps, freq='W-MON')[1:]
        forecast_df = pd.DataFrame(forecast, index=forecast_dates)

        # Combine the original data with the forecast for the final output
        itemlevelpred['ARIMA'] = model_fit.fittedvalues
        combined_df = pd.concat([itemlevelpred, forecast_df], axis=0)
        
        # Final prediction output
        return combined_df
