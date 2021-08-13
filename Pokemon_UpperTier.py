###Based on clustering, I'm afraid that there's just a big pool of garbage Pokemon that end up in a general "low tier" group
###Here I'm filtering out some of the junk and just working with "playable" Pokemon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA

data = pd.read_csv('pokedex_(Update_04.21).csv')
print(data.count)
print(list(data.columns))

#Removing Megas, Galarians, and Alolans from data - should be a more elegant way to do this
#Previously did this after creating upper tier, but have since done this before splitting data
data = data[~data['name'].str.contains('Mega')]
data = data[~data['name'].str.contains('Alolan')]
data = data[~data['name'].str.contains('Galarian')]
print(data.count)

desc = [('Mean', data['total_points'].mean()), ('StDev', data['total_points'].std()), ('Max Val', data['total_points'].max()), ('Min Val', data['total_points'].min()), ('Median', data['total_points'].median())]
print(desc)
#Mean is 431.23, high variance (stdev = 118.02), median = 450.0
#Thinking median split is the way to go
med = data['total_points'].median()
df = data[data['total_points'] >= med]
print(data.count, df.count)

print(df.head())
desc_tt = [('Mean', df['total_points'].mean()), ('StDev', df['total_points'].std()), ('Max Val', df['total_points'].max()), ('Min Val', df['total_points'].min()), ('Median', df['total_points'].median())]
print(desc_tt)

for column in df[['generation', 'status', 'type_1']]:
    print(column, pd.crosstab(index=df[column], columns='count'))

###Visualizing ratios of each stat to total - nothing unexpected - will create manually below
for column in df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]:
    columnSeriesObj = df[column]
    df['column_rat'] = columnSeriesObj / df['total_points']
    print(column, df['column_rat'].mean())

df['hp_rat'] = df['hp'] / df['total_points']
df['attack_rat'] = df['attack'] / df['total_points']
df['defense_rat'] = df['defense'] / df['total_points']
df['sp_atk_rat'] = df['sp_attack'] / df['total_points']
df['sp_def_rat'] = df['sp_defense'] / df['total_points']
df['speed_rat'] = df['speed'] / df['total_points']

###Re-run classification algorithms on this dataset, and then really investigate differences between clusters
    #i.e. ANOVAs, MANOVAs, chi-squares, correlation matrices, differences in effect sizes, variance and stds
##K-means 
from sklearn.cluster import KMeans
x = df.iloc[:, -6:].values

print(x)

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


#Settled on 4 clusters based on elbow graph (5 looked better when dealing with raw values v ratios)
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
kmeans.fit(x)
y_kmeans = kmeans.fit_predict(x)
df['cluster_km'] = y_kmeans

#This is one way of comparing means across clusters, but a better way is using pd.pivot_table to put it all in one
#single list -- hence why below is commented out and pd.pivot_table is not
"""for column in df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']]:
    grouped_df = df.groupby('cluster_km')
    mean_grouped_col_km = grouped_df[column].mean()
    print(column, mean_grouped_col_km)"""

#Anovas
for column in df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']]:
    columnSeriesObj = df[column]
    model = ols('columnSeriesObj.values ~ cluster_km', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('ANOVA', column, anova_table)

print('MEAN', pd.pivot_table(df, index=df['cluster_km'], 
    values=df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']], 
    aggfunc='mean'))

print('STANDARD DEV', pd.pivot_table(df, index=df['cluster_km'], 
    values=df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']], 
    aggfunc='std'))

#MANOVA across ratios from k-means clusters
maov_km = MANOVA.from_formula('hp_rat + attack_rat + defense_rat + sp_atk_rat + sp_def_rat + speed_rat ~ cluster_km', data=df)
print(maov_km.mv_test())

#Visualizing clusters
"""
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
plt.show()"""

##Hierarchical clustering - didn't love results from k-means
'''import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward', ))
plt.title('Dendrogram')
plt.xlabel('Pokemon (Rows)')
plt.ylabel('Euclidean Distances (b/w rows)')
plt.show()'''

#I like 6-clusters here based on dendrogram, but didn't yield any real differences between groups
#going with 4 to compare against k-means
from sklearn.cluster import AgglomerativeClustering
y_hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')

y_hc = y_hc.fit_predict(x)

df['cluster_hc'] = y_hc

for column in df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']]:
    columnSeriesObj = df[column]
    model = ols('columnSeriesObj.values ~ cluster_hc', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('ANOVA', column, anova_table)

print('MEAN', pd.pivot_table(df, index=df['cluster_hc'], 
    values=df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']], 
    aggfunc='mean'))

print('STANDARD DEV', pd.pivot_table(df, index=df['cluster_hc'], 
    values=df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']], 
    aggfunc='std'))

#MANOVA across ratios for hierarchical clusters
maov_hc = MANOVA.from_formula('hp_rat + attack_rat + defense_rat + sp_atk_rat + sp_def_rat + speed_rat ~ cluster_km', data=df)
print(maov_hc.mv_test())

#Visualizing clusters -- not helpful since it's 6D
'''
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(x[y_hc == 5, 0], x[y_hc == 5, 1], s = 100, c = 'magenta', label = 'Cluster 6')
plt.title('Clusters of Pokemon')
plt.xlabel('Multivariate score')
plt.ylabel('Multivariate score')
plt.legend()
plt.show()'''

"""for column in df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']]:
    grouped_df = df.groupby('cluster_hc')
    mean_grouped_col_hc = grouped_df[column].mean()
    print(column, mean_grouped_col_hc)
"""

###other thought - calculate total 'against' score to include in potential team building metrics? Would want to know what is strongest agains most types
###Also need to figure out how to best integrate types in order to have a balanced team
###Would it be worthwile to turn the score data into orindal ranks? Could then see who ranks best in each, sum ranks across, and see who comes out on top
###Re-run classification algorithms on this dataset, and then really investigate differences between clusters
    #i.e. ANOVAs, MANOVAs, chi-squares, correlation matrices, differences in effect sizes, variance and stds

