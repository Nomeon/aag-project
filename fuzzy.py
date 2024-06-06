import dask.dataframe as dd
from rapidfuzz import process
import pandas as pd
import pgeocode as pgeo

nomi = pgeo.Nominatim('de')

df = pd.read_parquet('Final.parquet')
df['PostalCode'] = df['PostalCode'].astype(str)
df = df.dropna(subset=["City", "PostalCode"])
df = df[df['PostalCode'].str.match(r"^(?!00000)\d{5}$")]

print(df.info())
print('Parquet file loaded')

plz = pd.read_csv('data/German-Zip-Codes.csv', dtype={'Plz': str}, sep=';', low_memory=False, encoding='utf-8-sig')
plz = plz[['Ort', 'Plz', 'Bundesland']]
plz['Plz'] = plz['Plz'].astype(str)
plz = plz.rename(columns={
    "Ort": "City",
    "Plz": "PostalCode",
    "Bundesland": "State"
  }
)

print('PLZ file loaded')

merged_df = pd.merge(df, plz, on='PostalCode', how='left', suffixes=('_inconsistent', '_correct'))
merged_df['City_correct'] = merged_df['City_correct'].fillna(merged_df['City_inconsistent'])

print('Dataframes merged')

# Create a dictionary with postal codes as keys and a list of possible cities as values
postal_code_to_cities = merged_df.drop_duplicates(subset=['PostalCode', 'City_correct']).groupby('PostalCode')['City_correct'].unique().to_dict()

def get_closest_match(row):
    possible_cities = postal_code_to_cities.get(row['PostalCode'], [])
    if len(possible_cities) == 0:
        return row['City_inconsistent']  # Return original if no cities are found
    closest_match = process.extractOne(row['City_inconsistent'], possible_cities)[0]
    return closest_match

dask_df = dd.from_pandas(merged_df, npartitions=10)

dask_df['Final_City'] = dask_df.map_partitions(
    lambda df: df.apply(lambda row: get_closest_match(row), axis=1),
    meta=('Final_City', 'object')  # Define the expected return type
)

merged_df = dask_df.compute()


print('Closest matches found')

def find_nearest_state(postal_code):
    original_code = postal_code
    offset = 1  # Starting offset
    
    while True:
        decrementing_code = postal_code - offset
        incrementing_code = postal_code + offset

        # Check decrementing postal code if it doesn't go negative
        if decrementing_code >= 0:
            location = nomi.query_postal_code(str(decrementing_code))
            if not pd.isna(location['state_name']):
                print(f"Found state {location['state_name']} for postal code {original_code}, with searched postal code {decrementing_code}")
                return location['state_name']

        # Check incrementing postal code
        location = nomi.query_postal_code(str(incrementing_code))
        if not pd.isna(location['state_name']):
            print(f"Found state {location['state_name']} for postal code {original_code}, with searched postal code {incrementing_code}")
            return location['state_name']

        # Increase the offset
        offset += 1

        # Optionally add a breaking condition if the offset gets too large
        if offset > 200:  # Assuming you don't want to go beyond 200 postal codes away
            print(f"No valid state found near postal code {original_code}")
            return None

def fill_missing_states(row):
    if pd.isna(row['State']):
        postal_code = int(row['PostalCode'])  
        location = nomi.query_postal_code(str(postal_code))
        if pd.isna(location['state_name']):
            return find_nearest_state(postal_code)
        else:
            return location['state_name']
    else:
        return row['State']

# Apply the function to fill missing states
merged_df['State'] = merged_df.apply(fill_missing_states, axis=1)

print('States filled')

# Filter out duplicates
filtered_df = merged_df.drop_duplicates(subset=['ArticleID', 'OrderDate', 'CustomerID'])
filtered_df.loc[:, 'Final_City'] = filtered_df['Final_City'].str.split(',').str[0]

print('Duplicates filtered')

# filtered_df.to_parquet('Final_Fuzzy.parquet', index=False)

print(filtered_df.info())