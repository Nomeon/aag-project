import dask.dataframe as dd
from rapidfuzz import process
import pandas as pd
import pgeocode as pgeo


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