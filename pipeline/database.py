import os
import glob
import sqlite3
import numpy as np
import pandas as pd


def process_lstm_files(file_pattern):
    """Reads and processes LSTM files matching a pattern.
    
    Args:
        file_pattern (str): The file pattern to match.
        
    Returns:
        pd.DataFrame: The combined DataFrame.
    """
    csv_files = glob.glob(file_pattern)
    all_data = []

    for file in csv_files:
        df = pd.read_csv(file)
        cluster_name = '1' if 'Cluster1' in file else '2' if 'Cluster2' in file else '3' if 'Cluster3' in file else 'Unknown'
        df['Cluster'] = cluster_name
        df.loc[df['Revenue'] == df['Predict'], 'Revenue'] = np.nan  # Remove Revenue if it equals Predict
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    total_predict_per_cluster = combined_df.groupby('Cluster')['Predict'].sum()
    for cluster in ['1', '2', '3']:
        print(f"Total Predict Value for Cluster {cluster}: {total_predict_per_cluster.get(cluster, 'No Data')}")
    
    return combined_df


def read_and_process_excel(file_pattern, columns_to_keep, extract_item_type=False):
    """Reads and processes Excel files matching a pattern, with optional extraction of item type.

    Args:
        file_pattern (str): The file pattern to match.
        columns_to_keep (list): The columns to keep.
        extract_item_type (bool): Whether to extract item type and location.

    Returns:
        pd.DataFrame: The combined DataFrame.
    """
    xlsx_files = glob.glob(file_pattern)
    all_data = []

    for file in xlsx_files:
        if 'predictions' in file.lower():
            print(f"\nProcessing file: {file}")
            try:
                df = pd.read_excel(file, engine='openpyxl')
                print(f"Successfully read {file}.")
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

            # Extract item type and location based on the file type
            base_name = os.path.basename(file).replace('.xlsx', '').replace('predictions', '').replace('2024', '').strip()

            if extract_item_type:
                # For specific product predictions, split item type and location
                item_type = ''.join(filter(str.isalpha, base_name.split('Bayern')[0].split('Brandenburg')[0]))  # ItemType part
                location_name = base_name.replace(item_type, '').strip()  # Location part
                df['ItemType'] = item_type
                df['Location'] = location_name
            else:
                # For general predictions, location is the only part
                df['Location'] = base_name

            # Keep only relevant columns
            df = df[columns_to_keep]

            # Ensure numeric columns are floats and clean NaNs
            numeric_cols = ['Quantity', 'Train', 'Test', 'Predict']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            # Remove Quantity values where Quantity == Predict
            df.loc[df['Predict'] == df['Quantity'], 'Quantity'] = pd.NA

            # Handle NaNs and rounding
            df[numeric_cols] = df[numeric_cols].fillna(0).round(2).replace(0, pd.NA)

            # Add to data list
            all_data.append(df)
        else:
            print(f"Skipping file: {file} (does not contain 'predictions')")

    # Combine all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    print("\nCombined DataFrame (first few rows):")
    return combined_df


def save_to_sqlite(df, db_name, table_name):
    """Saves a DataFrame to SQLite.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        db_name (str): The database name.
        table_name (str): The table name.
    """

    with sqlite3.connect(db_name) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        df.to_sql(table_name, conn, if_exists='append', index=False)
        df_preview = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(f"\nPreview of table '{table_name}':\n", df_preview)
