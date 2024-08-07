import dask.dataframe as dd
from rapidfuzz import process
import pandas as pd
import pgeocode as pgeo

postalcode_cache = {}
seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}

def convertDataToParquet(df: pd.DataFrame, name: str) -> None:
  df.to_parquet(name, index=False)


def loadData(rechnung_path: str, kunden_path: str) -> tuple:
  df_rechnung = pd.read_parquet(rechnung_path)
  df_rechnung['Kunde_Verkauf_SK'] = df_rechnung['Kunde_Verkauf_SK'].astype(str) # Convert to string for merging
  df_kunden = pd.read_csv(kunden_path, sep=';', low_memory=False, encoding='utf-8-sig', dtype={'Kunde_SK': str, 'PLZ-Code': str})
  return df_rechnung, df_kunden


def mergeOnKunden(df_rechnung: pd.DataFrame, df_kunden: pd.DataFrame) -> pd.DataFrame:
  rechnung_columns = ["Belegnummer", "Unternehmen", "Artikel_SK", "Auftragsdatum_SK", "Kunde_Verkauf_SK", "Umsatztyp", "Preis Verpackungseinheit", "Menge", "Nettoumsatz", "Productgroup", "Productsubgroup", "Business Area", "Type"]
  df_rechnung = df_rechnung[rechnung_columns]

  kunden_columns = ["Kunde_SK", "Ort", "PLZ-Code", "Branchengruppe", "Vertriebskanalkategorie", "Vertriebskanal"]
  df_kunden = df_kunden[kunden_columns]
  df_kunden = df_kunden.rename(columns={"Kunde_SK": "Kunde_Verkauf_SK"})

  df = pd.merge(df_rechnung, df_kunden, on="Kunde_Verkauf_SK")
  return df


def initialCleaning(df: pd.DataFrame) -> pd.DataFrame:

  # Translate to English
  df = df.rename(columns={
    "Belegnummer": "OrderNumber",
    "Artikel_SK": "ArticleID",
    "Unternehmen": "Company",
    "Auftragsdatum_SK": "OrderDate",
    "Kunde_Verkauf_SK": "CustomerID",
    "Umsatztyp": "RevenueType",
    "Preis Verpackungseinheit": "PricePackagingUnit",
    "Menge": "Quantity",
    "Nettoumsatz": "NetRevenue",
    "Productgroup": "ProductGroup",
    "Productsubgroup": "ProductSubgroup",
    "Business Area": "BusinessArea",
    "Type": "Type",
    "Ort": "City",
    "Vertriebskanalkategorie": "SalesChannelCategory",
    "Vertriebskanal": "SalesChannel",
    "PLZ-Code": "PostalCode",
    "Branchengruppe": "IndustryGroup"
    }
  )

  df = df.dropna(subset=["City", "PostalCode"])

  # Filter PostalCode without using lookahead regex
  df = df[df['PostalCode'].str.len() == 5]
  df = df[df['PostalCode'] != '00000']

  df['OrderDate'] = pd.to_datetime(df['OrderDate'], format='%Y%m%d')
  
  # More optimal season calculation
  df['Season'] = ((df['OrderDate'].dt.month % 12 + 2) // 3 % 4 + 1).map(seasons)
  
  df['SalesChannel'] = df['SalesChannelCategory'].str.split('_').str[0]
  df = df[df['SalesChannel'].isin(['B2B', 'B2C', 'B2B2C'])]

  # Remove rows where NetRevenue is NaN
  df = df.dropna(subset=['NetRevenue'])
  return df


def blendPostalCodes(df: pd.DataFrame, plz_path: str):
  df['PostalCode'] = df['PostalCode'].astype(int)
  df_plz = pd.read_csv(plz_path, dtype={'plz': int}, sep=',', low_memory=False, encoding='utf-8-sig')
  df_plz = df_plz[['ort', 'plz', 'bundesland']]
  df_plz = df_plz.rename(columns={
    "ort": "City",
    "plz": "PostalCode",
    "bundesland": "State"
  })
  merged_df = pd.merge(df, df_plz, on='PostalCode', how='left', suffixes=('_inconsistent', '_correct'))
  merged_df['City_correct'] = merged_df['City_correct'].fillna(merged_df['City_inconsistent'])
  
  # Create a dictionary with postal codes as keys and a list of possible cities as values
  postal_code_to_cities = merged_df.drop_duplicates(subset=['PostalCode', 'City_correct']).groupby('PostalCode')['City_correct'].unique().to_dict()
  
  # Convert to Dask DataFrame for parallel processing
  dask_df = dd.from_pandas(merged_df, npartitions=10)

  dask_df['Final_City'] = dask_df.map_partitions(
    lambda df: df.apply(lambda row: getClosestMatch(row, postal_code_to_cities), axis=1),
    meta=('Final_City', 'object')  # Define the expected return type
  )

  merged_df = dask_df.compute()

  print('Closest matches found')

  merged_df['State'] = merged_df.apply(fillMissingStates, axis=1)
  merged_df['PostalCode'] = merged_df['PostalCode'].astype(str)

  print('Missing states filled in')

  filtered_df = merged_df.drop_duplicates(subset=['OrderNumber', 'ArticleID', 'OrderDate', 'CustomerID'])
  filtered_df.loc[:, 'Final_City'] = filtered_df['Final_City'].str.split(',').str[0]
  return filtered_df


def getClosestMatch(row: pd.Series, postal_code_to_cities: dict) -> str:
    possible_cities = postal_code_to_cities.get(row['PostalCode'], [])
    if len(possible_cities) == 0:
        return row['City_inconsistent']  # Return original if no cities are found
    closest_match = process.extractOne(row['City_inconsistent'], possible_cities)[0] #! INVESTIGATE
    return closest_match


def findNearestState(postal_code):
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


def fillMissingStates(row):
    if pd.isna(row['State']):
        postal_code = row['PostalCode']  
        location = nomi.query_postal_code(postal_code)
        if pd.isna(location['state_name']):
            return findNearestState(postal_code)
        else:
            return location['state_name']
    else:
        return row['State']


def addFeatures(df: pd.DataFrame) -> pd.DataFrame:
    reference_date = pd.to_datetime('2023-12-31', format='%Y%m%d')
    df['DaysSinceLastOrder'] = (reference_date - df['OrderDate']).dt.days


def finalCleaning(df: pd.DataFrame) -> pd.DataFrame:
  # Manually fill in missing states
  df.loc[(df['PostalCode'] == '69966') & (df['State'].isnull()), 'State'] = 'Baden-Württemberg'
  df.loc[(df['PostalCode'] == '8312') & (df['State'].isnull()), 'State'] = 'Sachsen'
  df.loc[(df['PostalCode'] == '40002') & (df['State'].isnull()), 'State'] = 'Nordrhein-Westfalen'
  df.loc[(df['PostalCode'] == '7801') & (df['State'].isnull()), 'State'] = 'Thüringen'

  # Drop City_inconsistent, City_correct, and rename Final_City to City
  df = df.drop(columns=['City_inconsistent', 'City_correct'])
  df = df.rename(columns={'Final_City': 'City'})
  return df


if __name__ == "__main__":
  nomi = pgeo.Nominatim('de')
  df_rechnung, df_kunden = loadData('data/Rechnungen.parquet', 'data/Kunden.csv')
  print('Data loaded')
  df_merged = mergeOnKunden(df_rechnung, df_kunden)
  print('Data merged')
  df_cleaned = initialCleaning(df_merged)
  print('Data cleaned')  
  df_blended = blendPostalCodes(df_cleaned, 'data/zuordnung_plz_ort.csv')
  print('Data blended')
  df_final = finalCleaning(df_blended)
  df_final.to_parquet('data/Final.parquet', index=False)
