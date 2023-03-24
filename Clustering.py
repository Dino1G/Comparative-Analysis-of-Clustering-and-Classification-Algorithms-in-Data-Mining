#!/usr/bin/env python
# coding: utf-8

# # SCC403 Dataset 1

# ### 1. Import library & data

# In[1]:


get_ipython().run_cell_magic('capture', '', 'import pandas as pd\nimport numpy as np  \nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn import preprocessing   \nfrom sklearn.cluster import KMeans\nfrom sklearn.decomposition import PCA\nfrom sklearn.cluster import DBSCAN')


# In[2]:


colnames = ['Temperature (Min)','Temperature (Max)','Temperature (Mean)','Relative Humidity (Min)',
'Relative Humidity (Max)','Relative Humidity (Mean)','Sea Level Pressure (Min)','Sea Level Pressure (Max)',
'Sea Level Pressure (Mean)','Precipitation Total','Snowfall Amount','Sunshine Duration','Wind Gust (Min)',
'Wind Gust (Madf_pca)','Wind Gust (Mean)','Wind Speed (Min)','Wind Speed (Max)','Wind Speed (Mean)']
df = pd.read_csv('Data Files/ClimateDataBasel.csv', names=colnames, header=None)


# ### 2. Data pre-processing

# After checking the data, there are no missing values and no obvious outlier

# In[3]:


df.info()
df.describe().T


# Several of these features have collinearity

# In[4]:


fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", linewidths=0.3)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Corr Between Features")
#plt.savefig("Plots/corr.png")


# Normalization

# In[5]:


df_scale = preprocessing.MinMaxScaler().fit_transform(df)
df_scale.shape


# ### 3.Dimensionality reduction with PCA

# To get more than 80% of variance explained I need 3 principal components.

# In[6]:


pca = PCA()
pca.fit(df_scale)
plt.plot(range(1,19), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title("Explained Variance by Component(Clustering)")
plt.xlabel("Number of Component")
plt.ylabel("Cumlative Explained Variance")


# In[7]:


pca = PCA(n_components=3)
pca.fit(df_scale)
df_pca = pca.transform(df_scale)
df_pca = pd.DataFrame(df_pca,columns=['Component1','Component2','Component3'])


# In[8]:


plt.figure(figsize=(10,10))
var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
plt.bar(x=range(1,len(var)+1), height = var, tick_label = df_pca.columns)
plt.title("Explained Variance by Components")
plt.ylabel("Explained Variance(%)")
plt.show()


# In[9]:


from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(df_pca.iloc[:, 0], df_pca.iloc[:, 1], df_pca.iloc[:, 2])
plt.title("Data Distribution")
ax.set_xlabel('Component1')
ax.set_ylabel('Component2')
ax.set_zlabel('Component3')
plt.show()


# ### 4. Model training

# K-means

# In[10]:


get_ipython().run_cell_magic('capture', '', "wcss = []\nfor i in range(1,21):\n    kmeans_pca = KMeans(n_clusters=i, init='k-means++',random_state=42)\n    kmeans_pca.fit(df_pca)\n    wcss.append(kmeans_pca.inertia_)")


# In[11]:


plt.plot(range(1,21),wcss, marker='o')
plt.title("K-means Cluster Decision")
plt.ylabel("WCSS")
plt.xlabel("Number of Cluster")


# The kink comes at the 4 clusters mark. So, we’ll be keeping a four-cluster solution.

# In[12]:


get_ipython().run_cell_magic('capture', '', "kmeans_pca = KMeans(n_clusters=4, init='k-means++', random_state=42)\nkmeans_pca.fit(df_pca)")


# In[13]:


df_pk = pd.concat([df.reset_index(drop=True),pd.DataFrame(df_pca)],axis=1)
df_pk.columns.values[-3:] = ['Component1','Component2','Component3']
df_pk['Labels'] = kmeans_pca.labels_
df_pk.head()


# In[14]:


sns.scatterplot(data=df_pk, x='Component1',y='Component2',hue='Labels')
plt.title("Clustering Results")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title("Clustering Results")

for s in df_pk.Labels.unique():
    ax.scatter(df_pk.Component1[df_pk.Labels==s],df_pk.Component2[df_pk.Labels==s],df_pk.Component3[df_pk.Labels==s],label=s)

ax.legend()


# DBScan

# In[15]:


from sklearn.neighbors import NearestNeighbors

nearest_neighbors = NearestNeighbors(n_neighbors=6)
neighbors = nearest_neighbors.fit(df_pca)

distances, indices = neighbors.kneighbors(df_pca)
distances = np.sort(distances[:,5], axis=0)


# In[16]:


from kneed import KneeLocator

i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")

print(distances[knee.knee])


# The knee occurs at approximately 0.14
# For multidimensional dataset, minPts should be 2 * number of dimensions

# In[17]:


clusters = DBSCAN(eps=0.14, min_samples=6).fit(df_pca)

p = sns.scatterplot(data=df_pca, x=df_pca.iloc[:,0], y=df_pca.iloc[:,1], hue=clusters.labels_, legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.2), title='Clusters')
plt.title("Clustering Results (noisy points -1 cluster)")
plt.show()


df_test = pd.concat([df_pca.reset_index(drop=True),pd.DataFrame(clusters.labels_,columns=['Labels'])],axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for s in df_test.Labels.unique():
    ax.scatter(df_test.Component1[df_test.Labels==s],df_test.Component2[df_test.Labels==s],df_test.Component3[df_test.Labels==s],label=s)

plt.title("Clustering Results (noisy points -1 cluster)")
ax.legend()


# In[ ]:




