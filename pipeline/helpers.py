import pandas as pd

def loadInitialData(rechnung_path: str, kunden_path: str) -> tuple:
  """Loads the initial data from AAG and returns a DataFrame for Kunden and Rechnungen.

  Args:
    rechnung_path (str): The path to the Rechnungen data.
    kunden_path (str): The path to the Kunden data.

  Returns:
    tuple: A tuple containing the Rechnungen and Kunden DataFrames.
  """
  df_rechung = pd.read_parquet(rechnung_path)
  df_rechung['Kunde_Verkauf_SK'] = df_rechung['Kunde_Verkauf_SK'].astype(str)
  df_kunden = pd.read_csv(kunden_path, sep=';', low_memory=False, encoding='utf-8-sig', dtype={'Kunde_SK': str, 'PLZ-Code': str})
  return df_rechung, df_kunden


def loadParquetFile(path: str) -> pd.DataFrame:
  """Loads a parquet file from the given path and returns it as a pandas DataFrame.

  Args:
    path (str): The path to the parquet file.

  Returns:
    pd.DataFrame: The parquet file as a pandas DataFrame.
  """
  return pd.read_parquet(path)


def convertDataToParquet(df: pd.DataFrame, name: str) -> None:
  """Converts the given DataFrame to a parquet file with the given name.

  Args:
    df (pd.DataFrame): The DataFrame to convert.
    name (str): The name of the parquet file.
  """
  df.to_parquet(name, index=False)