import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('pokedex_(Update_04.21).csv')

#%%
###K-means - I'm not sure I love this one
from sklearn.cluster import KMeans
x = data.iloc[:, 18:24].values

print(x.shape)

wcss = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
print(wcss)

"""plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()"""

#Was between 5 and 6 but ultimately settled on 5 - based (partly) on dendrogram and that it looked like group 2 was just a group of awful Pokemon
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
kmeans.fit(x)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)

data['cluster_km'] = y_kmeans
print(data.head())

for column in data[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]:
    grouped_df = data.groupby('cluster_km')
    mean_grouped_col_km = grouped_df[column].mean()
    print(column, mean_grouped_col_km)

#Visualizing clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 15, c = 'red', label = 'cluster_km 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 15, c = 'blue', label = 'cluster_km 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 15, c = 'green', label = 'cluster_km 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 15, c = 'cyan', label = 'cluster_km 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 15, c = 'magenta', label = 'cluster_km 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('cluster_km of Pokemon')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show

"""
#%%
##Seeing if Naive Bayes can predict cluster membership based on the same input variables - feels kinda wild, but can get a sense of the variance within each cluster
#Splitting training from test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y_kmeans, test_size = 0.66, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

##Building and training naive bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

#Predicting results
y_pred_nb = classifier.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_nb.reshape(len(y_pred_nb),1), y_test.reshape(len(y_test),1)),1))

#Testing signif difference between predicted and actual - don't like that these are significantly different
from scipy.stats import chisquare
print(chisquare((np.concatenate((y_pred_nb.reshape(len(y_pred_nb),1), y_test.reshape(len(y_test),1)),1))))

#Calculating confusion matrix AND accuracy score - however, these two metrics look really good?
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_nb)
print(cm)

print(accuracy_score(y_test, y_pred_nb))

#%%
##Also gonna try random forrest while we're at it - though NB looks okay-ish
##Random Forest Classifier
#Import dataset
#Building and training model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
classifier.fit(x_train, y_train)

#Predicting results
y_pred_rf = classifier.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_rf.reshape(len(y_pred_rf),1), y_test.reshape(len(y_test),1)),1))

#Testing signif difference between predicted and actual
from scipy.stats import chisquare
print(chisquare((np.concatenate((y_pred_rf.reshape(len(y_pred_rf),1), y_test.reshape(len(y_test),1)),1))))

#Calculating confusion matrix AND accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)

print(accuracy_score(y_test, y_pred_rf))

#potentially interesting to see if there is any bias
cm_preds = confusion_matrix(y_pred_nb, y_pred_rf)
print(cm_preds)
"""

##Going to try Hierarchical Clustering and see how that comes out - and compare them against one another
#Using dendrogram to find optimal number of clusters
"""
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward', ))
plt.title('Dendrogram')
plt.xlabel('Customers (Rows)')
plt.ylabel('Euclidean Distances (b/w rows)')
plt.show()
"""

#Training hierarchical cluster model on data
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)
print(y_hc)

#Visualizing clusters
"""
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 10, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 10, c = 'magenta', label = 'Cluster 5')
plt.title('clusters_hc of Pokemon')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
"""

data['cluster_hc'] = y_hc

for column in data[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]:
    grouped_df = data.groupby('cluster_hc')
    mean_grouped_col_hc = grouped_df[column].mean()
    print(column, mean_grouped_col_hc)

##Repeating stat testing from k means
##Naive bayes
##Splitting training from test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y_kmeans, test_size = 0.66, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

##Building and training naive bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

#Predicting results
y_pred_nb = classifier.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_nb.reshape(len(y_pred_nb),1), y_test.reshape(len(y_test),1)),1))

#Testing signif difference between predicted and actual
from scipy.stats import chisquare
print(chisquare((np.concatenate((y_pred_nb.reshape(len(y_pred_nb),1), y_test.reshape(len(y_test),1)),1))))

#Calculating confusion matrix AND accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_nb)
print(cm)

print(accuracy_score(y_test, y_pred_nb))

###Random Forest Classifier
#Building and training model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
classifier.fit(x_train, y_train)

#Predicting results
y_pred_rf = classifier.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_rf.reshape(len(y_pred_rf),1), y_test.reshape(len(y_test),1)),1))

#Testing signif difference between predicted and actual
from scipy.stats import chisquare
print(chisquare((np.concatenate((y_pred_rf.reshape(len(y_pred_rf),1), y_test.reshape(len(y_test),1)),1))))

#Calculating confusion matrix AND accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)

print(accuracy_score(y_test, y_pred_rf))

#potentially interesting to see if there is any bias
cm_preds = confusion_matrix(y_pred_nb, y_pred_rf)
print(cm_preds)
