import helpers
import preprocessing
import customerClustering
import customerPredictions
import database
import pandas as pd
import pgeocode as pgeo


def preprocessData(rechnung_path: str, kunden_path: str, nomi) -> pd.DataFrame:
  """Runs the cleaning pipeline to convert the initial data into a cleaned parquet file.

  Args:
    rechnung_path (str): The path to the Rechnungen_new.parquet file.
    kunden_path (str): The path to the Kunden.csv file.

  Returns:
    pd.DataFrame: The cleaned DataFrame.
  """
  df_rechnung, df_kunden = helpers.loadInitialData(rechnung_path, kunden_path)
  print('Data loaded.')
  df = preprocessing.mergeOnKunden(df_rechnung, df_kunden)
  helpers.convertDataToParquet(df, "data/merged_data.parquet")
  print('Data merged.')
  df = preprocessing.initialCleaning(df)
  helpers.convertDataToParquet(df, "data/initial_cleaned_data.parquet")
  print('Data cleaned.')
  df = preprocessing.blendPostalCodes(df, 'data/zuordnung_plz_ort.csv', nomi)
  helpers.convertDataToParquet(df, "data/blended_data.parquet")
  print('Postal codes blended.')
  df = preprocessing.finalCleaning(df)
  helpers.convertDataToParquet(df, "data/cleaned_data.parquet")
  print('Data done.')
  return df


def customerPrediction(df: pd.DataFrame):
  """Runs the full customer prediction pipeline.

  Args:
    df (pd.DataFrame): The cleaned DataFrame.
  """
  
  customer2023 = df[df['OrderDate'].dt.year == 2023]
  top25df = customerClustering.getTop25PercentCustomers(customer2023)
  customerClusters = customerClustering.clusterRFM(top25df)
  print('Data clustered.')

  predictions = customerPredictions.predictRevenuePerCluster(customerClusters, df, "LSTM", 2,2)
  print('Predictions done.')
  print(predictions)


def main():
  """Runs the full pipeline including the predictions.

  """
  rechnung_path = "data/Rechnungen_new.parquet"
  kunden_path = "data/Kunden.csv"

  nomi = pgeo.Nominatim('de')

  df = preprocessData(rechnung_path, kunden_path, nomi)
  print('Data preprocessed.')

  customerPrediction(df)


  
  combined_lstm_df = database.process_lstm_files('data/LSTM/New/Option4/*.csv')
  database.save_to_sqlite(combined_lstm_df, 'predictions_LSTM.db', 'predictions_LSTM')

  columns_to_keep_general = ['Date', 'Quantity', 'Train', 'Test', 'Predict', 'Location']
  combined_df_general = database.read_and_process_excel('data/LSTM/Items/New/*.xlsx', columns_to_keep_general, extract_item_type=False)
  database.save_to_sqlite(combined_df_general, 'predictions_LSTM.db', 'predictions_LSTM_Items')

  columns_to_keep_products = ['Date', 'Quantity', 'Train', 'Test', 'Predict', 'ItemType', 'Location']
  combined_df_products = database.read_and_process_excel('data/LSTM/Items/ItemPredicting/*.xlsx', columns_to_keep_products, extract_item_type=True)
  database.save_to_sqlite(combined_df_products, 'predictions_LSTM.db', 'predictions_LSTM_Items_SpecificProducts')


if __name__ == '__main__':
  main()