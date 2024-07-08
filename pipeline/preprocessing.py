import dask.dataframe as dd
from rapidfuzz import process
import pandas as pd
import pgeocode as pgeo
import helpers


def mergeOnKunden(df_rechnung: pd.DataFrame, df_kunden: pd.DataFrame) -> pd.DataFrame:
  """Merges the Rechnungen and Kunden DataFrames on the Kunde_Verkauf_SK column.

  Args:
    df_rechnung (pd.DataFrame): The Rechnungen DataFrame.
    df_kunden (pd.DataFrame): The Kunden DataFrame.

  Returns:
    pd.DataFrame: The merged DataFrame.
  """

  rechnung_columns = ["Belegnummer", "Unternehmen", "Artikel_SK", "Auftragsdatum_SK", "Kunde_Verkauf_SK", "Umsatztyp", "Preis Verpackungseinheit", "Menge", "Nettoumsatz", "Productgroup", "Productsubgroup", "Business Area", "Type"]
  df_rechnung = df_rechnung[rechnung_columns]

  kunden_columns = ["Kunde_SK", "Ort", "PLZ-Code", "Branchengruppe", "Vertriebskanalkategorie", "Vertriebskanal"]
  df_kunden = df_kunden[kunden_columns]
  df_kunden = df_kunden.rename(columns={"Kunde_SK": "Kunde_Verkauf_SK"})

  df = pd.merge(df_rechnung, df_kunden, on="Kunde_Verkauf_SK")
  return df


def initialCleaning(df: pd.DataFrame) -> pd.DataFrame:
  """Cleans the DataFrame by renaming columns, dropping rows with missing values, filtering PostalCode, converting OrderDate to datetime, and adding a Season column.	

  Args:
    df (pd.DataFrame): The DataFrame to clean.

  Returns:
    pd.DataFrame: The cleaned DataFrame.
  """
  seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}

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

  df = df[df['PostalCode'].str.len() == 5]
  df = df[df['PostalCode'] != '00000']

  df['OrderDate'] = pd.to_datetime(df['OrderDate'], format='%Y%m%d')
  df['Season'] = ((df['OrderDate'].dt.month % 12 + 2) // 3 % 4 + 1).map(seasons)
  
  df['SalesChannel'] = df['SalesChannelCategory'].str.split('_').str[0]
  df = df[df['SalesChannel'].isin(['B2B', 'B2C', 'B2B2C'])]
  df = df.dropna(subset=['NetRevenue'])
  return df


def blendPostalCodes(df: pd.DataFrame, plz_path: str) -> pd.DataFrame:
  """Blends the PostalCode column with external Postalcodes data, to correct inconsistencies. source: https://www.suche-postleitzahl.org/ 

  Args:
    df (pd.DataFrame): The DataFrame to blend.
    plz_path (str): The path to the external PostalCodes data.

  Returns:
    pd.DataFrame: The blended DataFrame.
  """
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
    lambda df: df.apply(lambda row: helpers.getClosestMatch(row, postal_code_to_cities), axis=1),
    meta=('Final_City', 'object')  # Define the expected return type
  )
  merged_df = dask_df.compute()

  merged_df['State'] = merged_df.apply(helpers.fillMissingStates, axis=1)
  merged_df['PostalCode'] = merged_df['PostalCode'].astype(str)

  filtered_df = merged_df.drop_duplicates(subset=['OrderNumber', 'ArticleID', 'OrderDate', 'CustomerID'])
  filtered_df.loc[:, 'Final_City'] = filtered_df['Final_City'].str.split(',').str[0]
  return filtered_df


def addFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """Adds new features to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with the new features.
    """
    reference_date = pd.to_datetime('2023-12-31', format='%Y%m%d')
    df['DaysSinceLastOrder'] = (reference_date - df['OrderDate']).dt.days
    #! INCOMPLETE


def finalCleaning(df: pd.DataFrame) -> pd.DataFrame:
  """Cleans the DataFrame by filling in missing states and dropping unnecessary columns.

  Args:
    df (pd.DataFrame): The DataFrame to clean.

  Returns:
    pd.DataFrame: The cleaned DataFrame
  """
  # Manually fill in missing states
  df.loc[(df['PostalCode'] == '69966') & (df['State'].isnull()), 'State'] = 'Baden-Württemberg'
  df.loc[(df['PostalCode'] == '8312') & (df['State'].isnull()), 'State'] = 'Sachsen'
  df.loc[(df['PostalCode'] == '40002') & (df['State'].isnull()), 'State'] = 'Nordrhein-Westfalen'
  df.loc[(df['PostalCode'] == '7801') & (df['State'].isnull()), 'State'] = 'Thüringen'

  # Drop City_inconsistent, City_correct, and rename Final_City to City
  df = df.drop(columns=['City_inconsistent', 'City_correct'])
  df = df.rename(columns={'Final_City': 'City'})
  return df