import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score


def getDistributionCentres(row):
    """
    Assigns a Distribution Center based on the State values.

    Args:
        row (pd.Series): A row of data containing the 'State' field.

    Returns:
        str: The name of the Distribution Center corresponding to the state.
    """

    if row['State'] in ('Baden-Württemberg', 'Saarland'):
        ds = 'Baden-Württemberg'
    elif row['State'] in ('Nordrhein-Westfalen', 'Hessen', 'Niedersachsen', 'Bremen', 'Hamburg', 'Rheinland-Pfalz', 'Schleswig-Holstein'):
        ds = 'Nordrhein-Westfalen'
    elif row['State'] in ('Bayern', 'Sachsen' , 'Thüringen', 'Sachsen-Anhalt'):
        ds = 'Bayern'
    else:
        ds = 'Brandenburg'
    return ds



def getItemDataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the dataset for item-level clustering by filtering and calculating necessary metrics.

    Args:
        df (pd.DataFrame): The original DataFrame.

    Returns:
        pd.DataFrame: A DataFrame filtered and prepared for clustering, with Distribution Centers assigned.
    """

    # Keep only necessary columns
    df = df[['OrderNumber', 'OrderDate', 'Quantity', 'NetRevenue', 'ProductSubgroup', 'State']]
    # Assign DistributionCentres based on the State where an item was sold
    df['DistributionCenter'] = df.apply(getDistributionCentres, axis=1)
    # Fill empty Quantity values with 1.0 and keep only positive values
    df['Quantity'] = df['Quantity'].fillna(1.0)
    df = df[df['Quantity'] >= 1.0]
    # Exclude orders with dates in future
    df = df[df['OrderDate'] <= datetime.today().strftime('%Y%m%d')]
    
    # Calculate total revenue and total quantity
    totalRevenue = df['NetRevenue'].sum()
    totalQuantity = df['Quantity'].sum()

    # Group by ProductSubgroup and DistributionCenter and calculate share of total revenue plus share of total quantity
    SubGroupRevenue = df.groupby(['ProductSubgroup', 'DistributionCenter'])['NetRevenue'].sum().reset_index()
    SubGroupRevenue['shareOfRevenue'] = (SubGroupRevenue['NetRevenue']/totalRevenue)*100
    SubGroupRevenue = SubGroupRevenue.sort_values('shareOfRevenue', ascending=False)
    SubGroupRevenue['cumulatedShare'] = np.cumsum(SubGroupRevenue['shareOfRevenue'])

    SubGroupQuantity = df.groupby(['ProductSubgroup', 'DistributionCenter'])['Quantity'].sum().reset_index()
    SubGroupQuantity['shareOfQuantity'] = (SubGroupQuantity['Quantity']/totalQuantity)*100
    SubGroupQuantity = SubGroupQuantity.sort_values('shareOfQuantity', ascending=False)
    SubGroupQuantity['cumulatedQuantity'] = np.cumsum(SubGroupQuantity['shareOfQuantity'])

    # Merge two datasets on ProductSubgroup and DistributionCenter
    SubGroupRQ = pd.merge(SubGroupRevenue, SubGroupQuantity, on = ['ProductSubgroup', 'DistributionCenter'])
    SubGroupRQ = SubGroupRQ.sort_values('shareOfQuantity', ascending=False)

    # Get the data from the original dataset for all the ProductSubgroups in all DistributionCentres
    itemDataset = df[df['ProductSubgroup'].isin(SubGroupQuantity['ProductSubgroup'])]

    return itemDataset



def clusterRFC(itemDataset: pd.DataFrame) -> pd.DataFrame: 
    """
    Prepares the dataset for clustering and performs K-Means clustering on the item level.

    Args:
        item_dataset (pd.DataFrame): A DataFrame prepared for clustering.

    Returns:
        pd.DataFrame: A DataFrame with the RFC metrics and an additional column indicating the cluster assignment.
    """

    # Recency = time since last order, Frequency = number of orders, Capacity = sum of Quantity
    rfc = pd.DataFrame()

    # Timespan since most recent order per ProductSubgroup in a DistributionCenter
    reference_time = datetime.today().strftime('%Y%m%d')
    reference_date = pd.to_datetime(reference_time, format='%Y%m%d')
    # Calculating the difference in days from the reference date
    itemDataset['daysSinceLastOrder'] = (reference_date - itemDataset['OrderDate']).dt.days
    # Finding the most recent order in weeks for each ProductSubgroup in a DistributionCenter
    rfcrec = itemDataset.groupby(['ProductSubgroup', 'DistributionCenter'])['daysSinceLastOrder'].min().reset_index()

    # Counting of orders per ProductSubgroup in a DistributionCenter
    rfcfreq = itemDataset.groupby(['ProductSubgroup', 'DistributionCenter'])['OrderNumber'].count().reset_index()
    rfcfreq.columns = ['ProductSubgroup', 'DistributionCenter', 'orderCount']

    # Total quantity per ProductSubgroup in a DistributionCenter
    rfcmon = itemDataset.groupby(['ProductSubgroup', 'DistributionCenter'])['Quantity'].sum().reset_index()
    rfcmon.columns = ['ProductSubgroup', 'DistributionCenter', 'sumQuantity']

    # Merging features
    rfc1 = pd.merge(rfcrec, rfcfreq, on = ['ProductSubgroup', 'DistributionCenter'])
    rfc = pd.merge(rfc1, rfcmon, on = ['ProductSubgroup', 'DistributionCenter'])
    itemDataset = itemDataset.drop(['daysSinceLastOrder'], axis=1)

    rfc_clustering = rfc.drop(['ProductSubgroup', 'DistributionCenter'], axis=1)
    
    # Applying scaling
    robust_scaler = RobustScaler(quantile_range=(15, 85))

    scaled_rfc_np = robust_scaler.fit_transform(rfc_clustering)
    scaled_rfc = pd.DataFrame(scaled_rfc_np, columns = rfc_clustering.columns)

    # K-Means Clustering with Silhouette Score
    tot_withinss = []
    cluster_scores = []
    ks = list(range(3, 15))
    for k in ks:
        kmeans = KMeans(n_clusters = k, random_state = 1, n_init = 'auto') # Test MiniBatchKMeans 
        clusters = kmeans.fit_predict(scaled_rfc)
        score = silhouette_score(scaled_rfc, clusters)
        cluster_scores.append((k, score))
        print(f"Silhouette Score for {k} clusters: {score}")
        tot_withinss.append(kmeans.inertia_)
        

    # Getting the best cluster number
    best_cluster = max(cluster_scores, key=lambda x: x[1])[0]

    k_means_optimum = KMeans(n_clusters = best_cluster, random_state = 1, n_init = 'auto')
    k_means_optimum.fit_predict(scaled_rfc[["daysSinceLastOrder", "orderCount", "sumQuantity"]])
    rfc_clustering['Cluster_kmeans'] = k_means_optimum.labels_
    rfc_clustering['ProductSubgroup'] = rfc['ProductSubgroup']
    rfc_clustering['DistributionCenter'] = rfc['DistributionCenter']

    return rfc_clustering