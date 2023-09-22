#k-means clustering algorithm will produce labels that assign each company to different clusters
#k-means clustering aims to partition n observations into k clusters in which each observation 
#belongs to the cluster with the nearest mean, serving as a prototype of the cluster.

import yfinance as yf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime

# define instruments to download
companies_dict = {
"Amazon": "AMZN",
"Apple": "AAPL",
"Walgreen": "WBA",
"Broadcom": "AVGO",
"Boeing": "BA",
"Lockheed Martin":"LMT",
"McDonalds": "MCD",
"Intel": "INTC",
"IBM": "IBM",
"Texas Instruments": "TXN",
"MasterCard": "MA",
"Microsoft": "MSFT",
"General Electric": "GE",
"American Express": "AXP",
"Pepsi": "PEP",
"Coca Cola": "KO",
"Johnson & Johnson": "JNJ",
"Toyota": "TM",
"Honda": "HMC",
"Exxon": "XOM",
"Chevron": "CVX",
"Valero Energy": "VLO",
"Ford": "F",
"Bank of America": "BAC"
}



# Define the ticker symbol and date range
#ticker = "AAPL"
ticker = list(companies_dict.values())
start = datetime.datetime.now() - datetime.timedelta(days=5)
end = datetime.datetime.now()

# Download the historical stock prices
data = yf.download(ticker, start=start, end=end, interval='1h', auto_adjust=True)
panel_data = data.dropna()

# Print the data
#print(panel_data.head())

# Find Stock Open and Close Values
stock_close = data['Open']
stock_open = data['Close']

#print(stock_close.iloc[0])

# Calculate daily stock movement
stock_close = np.array(stock_close).T
stock_open = np.array(stock_open).T

row, col = stock_close.shape

# create movements dataset filled with 0's
movements = np.zeros([row, col])

for i in range(0, row):
    movements[i,:] = np.subtract(stock_close[i,:], stock_open[i,:])


for i in range(0, len(ticker)):
    print('Company: {}, Change: {}'.format(ticker[i][0], sum(movements[i][:])))

#import Normalizer
from sklearn.preprocessing import Normalizer
# create the Normalizer
normalizer = Normalizer()

new = normalizer.fit_transform(movements)

#print(new.max())
#print(new.min())
#print(new.mean())


plt.figure(figsize=(50,50))
#ax1 = plt.subplot(221)
#plt.plot(movements[0][:])
#plt.title(ticker[0])

#plt.subplot(222, sharey=ax1)
#plt.plot(movements[1][:])
#plt.title(ticker[1])

plt.hist(ticker, bins = 10)
plt.show()


# import machine learning libraries
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

# define normalizer
normalizer = Normalizer()

# create a K-means model with 10 clusters
kmeans = KMeans(n_clusters=4, max_iter=1000)

# make a pipeline chaining normalizer and kmeans
pipeline = make_pipeline(normalizer,kmeans)

# fit pipeline to daily stock movements
pipeline.fit(movements)
#Inertia is the sum of squared error for each cluster.
#Therefore the smaller the inertia the denser the cluster(closer together all the points are)

print(kmeans.inertia_)

# predict cluster labels
#predict() returns the label of each point in other words which cluster does the data point belong to
labels = pipeline.predict(movements)

# create a DataFrame aligning labels & companies
df = pd.DataFrame({'labels': labels, 'companies': ticker})

# display df sorted by cluster labels

print(df.sort_values('labels'))

#Principal Component Analysis (PCA)
#PCA is a dimensionality reduction algorithm that can be used to significantly speed up your
# unsupervised feature learning algorithm (in this case, K-means clustering) by reducing the
# number of features used in the learning algorithm whilst still keeping most of the variance of the data.

# PCA
from sklearn.decomposition import PCA 

# visualize the results
reduced_data = PCA(n_components = 2).fit_transform(new)

# run kmeans on reduced data
kmeans = KMeans(n_clusters=4)
kmeans.fit(reduced_data)
labels = kmeans.predict(reduced_data)

# create DataFrame aligning labels & companies
df = pd.DataFrame({'labels': labels, 'companies': ticker})

# Display df sorted by cluster labels
print(df.sort_values('labels'))


# Define step size of mesh
h = 0.01

# plot the decision boundary
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:,0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain abels for each point in the mesh using our trained model
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

# define colorplot
cmap = plt.cm.Paired

# plot figure
plt.clf()
plt.figure(figsize=(10,10))
plt.imshow(Z, interpolation='nearest',
 extent = (xx.min(), xx.max(), yy.min(), yy.max()),
 cmap = cmap,
 aspect = 'auto', origin='lower')
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=5)


# plot the centroid of each cluster as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
 marker='x', s=169, linewidth=3,
 color='w', zorder=10)

plt.title('K-Means Clustering on Stock Market Movements (PCA-Reduced Data)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

input("Press Enter to continue...")