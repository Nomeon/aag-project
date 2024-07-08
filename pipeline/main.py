import helpers
import preprocessing
import pandas as pd

def main():
  """Runs the cleaning pipeline to convert the initial data into a cleaned parquet file.

  """
  rechnung_path = "data/Rechnungen_new.parquet"
  kunden_path = "data/Kunden.csv"

  df_rechnung, df_kunden = helpers.loadInitialData(rechnung_path, kunden_path)
  print('Data loaded.')
  df = preprocessing.mergeOnKunden(df_rechnung, df_kunden)
  helpers.convertDataToParquet(df, "data/merged_data.parquet")
  print('Data merged.')
  df = preprocessing.initialCleaning(df)
  helpers.convertDataToParquet(df, "data/initial_cleaned_data.parquet")
  print('Data cleaned.')
  df = preprocessing.blendPostalCodes(df, 'data/zuordnung_plz_ort.csv')
  helpers.convertDataToParquet(df, "data/blended_data.parquet")
  print('Postal codes blended.')
  df = preprocessing.finalCleaning(df)
  helpers.convertDataToParquet(df, "data/cleaned_data.parquet")
  print('Data done.')


if __name__ == '__main__':
  main()