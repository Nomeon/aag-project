import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

# assumes complete preprocessingpipeline with added features 

# get top 25% most valuable customers
def getTop25PercentCustomers(df: pd.DataFrame) -> pd.DataFrame:
    """Takes the complete dataset and creates a subset of the top 25% of the most valuable customer based on their share of the total NetRevenue.

    Args:
      df (pd.DataFrame): The preprocessed DataFrame. Necessary Columns are "OrderNumber", "OrderDate", "CustomerID", "NetRevenue".

    Returns:
      pd.DataFrame: The subset DataFrame.
    """
    # keep only necessary columns
    df = df[["OrderNumber", "OrderDate", "CustomerID", "NetRevenue"]]
    
    # calculate total revenue
    totalRevenue = df['NetRevenue'].sum()

    # group by customer and calculate share of total revenue
    result = df.groupby('CustomerID')['NetRevenue'].sum().reset_index()
    result['shareOfRevenue'] = (result['NetRevenue']/totalRevenue)*100
    result = result.sort_values('shareOfRevenue', ascending=False)
    result["cumulatedShare"] = np.cumsum(result["shareOfRevenue"])

    # get the top 25% of the customers based on sorted share of revenue
    top25 =result.iloc[:(int(len(result)*(25/100)))]
    #print(top25.iloc[11193])
    # get the original data from the order dataset for the top 25% of the customers based on the CustomerID
    orders_top25 = df[df['CustomerID'].isin(top25["CustomerID"])]
    #print(orders_top25.info())

    return orders_top25


# create dataset on customer level containing behavior features & cluster customers

def clusterRFM(orders_top25: pd.DataFrame) -> pd.DataFrame: 
    """Takes the order dataset and creates RFM features per customer ID. 
    Then clusters the customers based on these features and for the optimal number of clusters between 3 and 10 based on Silhouette score. 
    Returns the customer dataset with assigned clusters.
  
    Args:
      orders_top25 (pd.DataFrame): The top25 percent of customers DataFrame.

    Returns:
      pd.DataFrame: The DataFrame including assigned clusters based on purchase behavior per CustomerID.
    """

    # Recency = time since last order, Frequency = number of orders, Monetary = sum of netRevenue
    rfm = pd.DataFrame()

    # timespan since most recent order per customer
    #reference_time = pd.Timestamp('2023-12-31') # static reference date if only one year in focus
    reference_time = datetime.today().strftime('%Y%m%d') # dynamic reference date if data about current orders
    reference_date = pd.to_datetime(reference_time, format='%Y%m%d')
    # Calculate the difference in days from the reference date
    orders_top25.loc[:, 'daysSinceLastOrder'] = (reference_date - orders_top25['OrderDate']).dt.days
    # Find the most recent order in weeks for each customerID
    rfmrec = orders_top25.groupby('CustomerID')['daysSinceLastOrder'].min().reset_index()

    # count of orders per customer
    rfmfreq = orders_top25.groupby('CustomerID')['OrderNumber'].nunique().reset_index()
    rfmfreq.columns = ['CustomerID', 'orderCount']

    # total revenue per customer
    rfmmon = orders_top25.groupby('CustomerID')['NetRevenue'].sum().reset_index()
    rfmmon.columns = ['CustomerID', 'sumRevenue']

    # merge features
    rfm1 = pd.merge(rfmrec, rfmfreq, on='CustomerID')
    rfm = pd.merge(rfm1, rfmmon, on='CustomerID')
    orders_top25 =orders_top25.drop(['daysSinceLastOrder'], axis=1)
    #print(rfm.head())
    #print(rfm.info())
    
    # encode data for clustering
    top25_df_encoded = rfm.drop(["CustomerID"], axis=1)

    # scale
    robust_scaler = RobustScaler(quantile_range=(15, 85))
    scaled_top25 = robust_scaler.fit_transform(top25_df_encoded)
    top25_scaled = pd.DataFrame(scaled_top25, columns=top25_df_encoded.columns)

    # apply K-Means Clustering with Silhouette Score, check for optimal number of k
    top25_tot_withinss = []
    top25_cluster_scores = []
    ks = list(range(3, 11))
    for k in ks:
#tbd test MiniBatchKMeans, delete random state?
        kmeans = KMeans(n_clusters=k, random_state=45, n_init="auto") 
        clusters = kmeans.fit_predict(top25_scaled)
        score = silhouette_score(top25_scaled, clusters)
        top25_cluster_scores.append((k, score))
    # Get best cluster number
    top25_best_cluster = max(top25_cluster_scores, key=lambda x: x[1])[0]
    #print(f"Best top25 cluster number based on Silhouette: {top25_best_cluster}")

    # add cluster assignment for optimal k to customer dataset 
# tbd delete random state?
    customer_k_means_optimum = KMeans(n_clusters=top25_best_cluster, random_state=45, n_init="auto") 
    customer_k_means_optimum.fit_predict(top25_scaled[["daysSinceLastOrder", "orderCount", "sumRevenue"]])
    top25_scaled['cluster'] = customer_k_means_optimum.labels_
    top25_scaled["CustomerID"] = rfm["CustomerID"]

# tbd create subsets of orders??
    # Create a dictionary to hold the subset DataFrames
        #cluster_subsets = {}
    # Loop through each unique cluster label and create a subset DataFrame
        #for cluster_num in top25_scaled['cluster'].unique():
        #    cluster_subsets[cluster_num] = top25_scaled[top25_scaled['cluster'] == cluster_num]   
        # get the specific subset for one cluster i:
        # subset = cluster_subsets[i]

    clustered_customers_df = top25_scaled[["CustomerID", "cluster"]]
    #print(top25_scaled.head())
    #print(clustered_customers_df.head())
    #print(top25_scaled['cluster'].value_counts())
    
    # return separated subsets: tbd adapt return type
    # return cluster_subsets
    return clustered_customers_df


###### ideally not needed with final preprocessing pipeline ######

# preprocessing to get final dataset with transformed values (cf. repo files)
#df = pd.read_parquet('cleaned_data.parquet')
#df.info()
#df = df[df['OrderDate'].dt.year == 2023]


##### function calls in main / pipeline: #####
#25percent = getTop25PercentCustomers(df)
#customerClusters = clusterRFM(25percent)

