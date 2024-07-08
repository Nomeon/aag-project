import pandas as pd
from rapidfuzz import process
import pgeocode as pgeo


postalcode_cache = {}
nomi = pgeo.Nominatim('de')


def loadInitialData(rechnung_path: str, kunden_path: str) -> tuple:
  """Loads the initial data from AAG and returns a DataFrame for Kunden and Rechnungen.

  Args:
    rechnung_path (str): The path to the Rechnungen data.
    kunden_path (str): The path to the Kunden data.

  Returns:
    tuple: A tuple containing the Rechnungen and Kunden DataFrames.
  """
  df_rechnung = pd.read_parquet(rechnung_path)
  df_rechnung['Kunde_Verkauf_SK'] = df_rechnung['Kunde_Verkauf_SK'].astype(str)
  df_kunden = pd.read_csv(kunden_path, sep=';', low_memory=False, encoding='utf-8-sig', dtype={'Kunde_SK': str, 'PLZ-Code': str})
  return df_rechnung, df_kunden


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


def getClosestMatch(row: pd.Series, postal_code_to_cities: dict) -> str:
    """Returns the closest match for the city based on the postal code, using fuzzy finding.

    Args:
        row (pd.Series): The row to process.
        postal_code_to_cities (dict): The dictionary containing postal codes as keys and a list of possible cities as values.

    Returns:
        str: The closest match for the city.
    """
    possible_cities = postal_code_to_cities.get(row['PostalCode'], [])
    if len(possible_cities) == 0:
        return row['City_inconsistent']  # Return original if no cities are found
    closest_match = process.extractOne(row['City_inconsistent'], possible_cities)[0] #! INVESTIGATE
    return closest_match


def fillMissingStates(row: pd.Series) -> str:
    """Fills the missing states in the DataFrame based on the PostalCode.

    Args:
        row (pd.Series): The row to process.

    Returns:
        str: The state name.
    """
    if pd.isna(row['State']):
        postal_code = row['PostalCode']  
        location = nomi.query_postal_code(postal_code)
        if pd.isna(location['state_name']):
            return findNearestState(postal_code)
        else:
            return location['state_name']
    else:
        return row['State']
    

def findNearestState(postal_code):
  """Finds the nearest state based on the given postal code.

  Args:
    postal_code (int): The postal code to find the nearest state for.

  Returns:
    str: The state name.
  """
  
  if postal_code in postalcode_cache:
    print(f"Found state {postalcode_cache[postal_code]} in cache for postal code {postal_code}")
    return postalcode_cache[postal_code]

  original_code = postal_code
  offset = 1
  
  while True:
    decrementing_code = postal_code - offset
    incrementing_code = postal_code + offset

    # Check decrementing postal code if it doesn't go negative
    if decrementing_code >= 0:
      location = nomi.query_postal_code(decrementing_code)
      if not pd.isna(location['state_name']):
        postalcode_cache[original_code] = location['state_name']
        print(f"Found state {location['state_name']} for postal code {original_code}")
        return location['state_name']

    # Check incrementing postal code
    location = nomi.query_postal_code(incrementing_code)
    if not pd.isna(location['state_name']):
      postalcode_cache[original_code] = location['state_name']
      print(f"Found state {location['state_name']} for postal code {original_code}")
      return location['state_name']

    # Increase the offset
    offset += 1

    # Optionally add a breaking condition if the offset gets too large
    if offset > 200:  # Assuming you don't want to go beyond 200 postal codes away
      print(f"No valid state found near postal code {original_code}")
      return None