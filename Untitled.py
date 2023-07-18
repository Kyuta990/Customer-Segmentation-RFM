#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install missingno


# In[2]:


pip install seaborn


# In[56]:


import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set the desired display options
pd.options.display.float_format = '{:.2f}'.format  # Show 2 decimal places


# In[4]:


# Provide the file path of the CSV file
file_path = 'online_retail_II.csv'

# Read the CSV file using Pandas
df = pd.read_csv(file_path)

# Print the head of the DataFrame
print(df.head())


# In[5]:


df.info()


# In[6]:


# Check for missing values in the entire DataFrame
missing_values = df.isnull().sum()

# Print the count of missing values for each column
print(missing_values)

# Check if any missing values exist in the DataFrame
if df.isnull().values.any():
    print("There are missing values in the dataset.")
else:
    print("There are no missing values in the dataset.")


# In[7]:


# Visualize missing values using a matrix
msno.matrix(df)

# Visualize missing values using a bar chart
msno.bar(df)

# Show the plots
plt.show()


# In[8]:


# Visualize missing values using a heatmap
sns.heatmap(df.isnull(), cbar=False)

# Show the plot
plt.show()


# In[10]:


print(df.duplicated().sum())


# In[11]:


df[df.duplicated()].head()


# In[12]:


df.loc[(df["Invoice"].astype(str) == '489517') & (df['StockCode'].astype(str) == '21912')] 


# In[13]:


df = df.drop(index=df[df.duplicated()].index)
print("The number of transaction after removing the duplicates: {0}".format(df.shape[0]))


# In[14]:


df.isnull().sum()


# In[15]:


df['StockCode'].describe()


# In[16]:


df['Quantity'].describe()


# In[17]:


df.loc[(df['Quantity'] == -8.099500e+04) | (df['Quantity'] == 8.099500e+04)]


# In[19]:


print("Cancelled invoices/transactions: {0}".format(df[df['Invoice'].astype(str).str[0] == 'C'].shape[0]))


# In[22]:


df['Price'].describe()


# In[23]:


df[df['Price'] < 0]


# In[24]:


df[df['Price'] > 10000]


# In[25]:


print('Transactions with zero unit price: {0}'.format(df[df['Price'] == 0].shape[0]))


# In[26]:


df[df['Price'] == 0].head()


# In[27]:


df['Description'].value_counts().sort_values(ascending=False)[:10]


# In[28]:


df['Country'].describe()


# In[29]:


df = df.drop(index = df[df['Customer ID'].isnull()].index)
print("Retail transactions after removing missing customer ids: {0}".format(df.shape[0]))


# In[30]:


import datetime as dt

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['Hour'] = df['InvoiceDate'].dt.hour


# In[35]:


hourly_sales = df[['Hour', 'Quantity']].groupby('Hour').sum()
hourly_sales.plot(kind="bar")
plt.figure(figsize=(8,6))
plt.title("Hourly Sales", fontsize=14)
sns.barplot(x = hourly_sales.index, y = hourly_sales['Quantity'])


# In[37]:


daily_sales = df[['Day', 'Quantity']].groupby('Day').sum()
plt.figure(figsize=(10,8))
plt.title("Daily Sales", fontsize=14)
sns.barplot(x = daily_sales.index, y = daily_sales["Quantity"])


# In[38]:


monthly_sales = df[['Month','Quantity']].groupby('Month').sum()
plt.figure(figsize=(8,6))
plt.title("Monthly Sales", fontsize=14)
sns.barplot(x = monthly_sales.index, y = monthly_sales['Quantity'])


# In[41]:


# Recency --> The freshness of customer purchase

# Calculate latest date from data set
max_date = df["InvoiceDate"].max()
# Calculate days passed since the last purchase
df['Days_passed'] = max_date - df["InvoiceDate"]
df["Days_passed"] = df['Days_passed'].dt.days
# Group Recency by customer id
recency = df[['Customer ID', 'Days_passed']].groupby('Customer ID').min()
recency.head(5)


# In[42]:


# Frequency of the customer transactions
frequency = df[['Customer ID', 'Invoice']].groupby('Customer ID').count()
frequency.head()


# In[43]:


# Monetory -> purchasing power of the customer
df['SaleAmount'] = df['Quantity'] * df['Price']
monetory = df[['Customer ID', 'SaleAmount']].groupby('Customer ID').sum()
monetory.head()


# In[44]:


# Merge recency, frequency and monetory dataframes
RFM = recency.merge(frequency, on='Customer ID').merge(monetory, on="Customer ID")
RFM = RFM.rename(columns={"Days_passed": 'Recency', 'Invoice': 'Frequency', 'SaleAmount': 'Monetory'})
RFM.head()


# # KMeans clustering

# In[47]:


range_n_clusters = [2,3,4,5,6,7,8]

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters = n_clusters, random_state=5)
    cluster_labels = clusterer.fit_predict(RFM)
    
    silhouette_avg = silhouette_score(RFM, cluster_labels)
    print("for n_clusters =", n_clusters, "Average Silhouette score = ", silhouette_avg)


# In[48]:


# Kmeans with number of clusters = 4
clusterer = KMeans(n_clusters=4, random_state=5)
cluster_labels = clusterer.fit_predict(RFM)


# In[49]:


RFM['Cluster'] = cluster_labels
RFM.groupby('Cluster').mean()


# Cluster 0 contains group of customers with low value of recency, frequency and monetory <br>
# Cluster 1 contains group of customers with high frequency value<br>
# Cluster 2 contains group of customers with high monetory value<br>
# Cluster 3 contains group of customers with low value of recency and moderate value of frequency and monetory

# # Principal component analysis (PCA)

# In[57]:


# Reduce dimension to 2 with PCA
pca = make_pipeline(StandardScaler(),
                   PCA(n_components = 2, random_state = 43))
RFM_transformed = pca.fit_transform(RFM)


# In[60]:


plt.figure(figsize=(10,8))
sns.scatterplot(x=RFM_transformed[:, 0], y=RFM_transformed[:, 1], hue=cluster_labels)
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title("Clusters", fontsize=14)

