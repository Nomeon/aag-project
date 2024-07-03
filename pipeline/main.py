import helpers
import preprocessing
import pandas as pd

def main():
  """Runs the cleaning pipeline to convert the initial data into a cleaned parquet file.

  """
  # rechnung_path = "data/Rechnungen.parquet"
  # kunden_path = "data/Kunden.csv"

  # df_rechnung, df_kunden = helpers.loadInitialData(rechnung_path, kunden_path)
  # df = preprocessing.mergeOnKunden(df_rechnung, df_kunden)
  # df = preprocessing.initialCleaning(df)
  # print(df.head())
  rechnung_path = "data/Rechnungen_new.csv"
  df = pd.read_csv(rechnung_path, sep=",", encoding="utf-8-sig", low_memory=False)
  print(df.head())
  print(df.info())
  print(df.columns)
  df.to_parquet("data/Rechnungen_new.parquet")


if __name__ == '__main__':
  main()