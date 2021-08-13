###Getting to know the data
#Importing all libraries used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.cluster.hierarchy as sch
from statsmodels.multivariate.manova import MANOVA



data = pd.read_csv('pokedex_(Update_04.21).csv')
print(len(data.index))
print(list(data.columns))

#Counts of categoricals
for column in data[['generation', 'status', 'species', 'type_1', 'type_2']]:
        print(pd.crosstab(index=data[column], columns = 'count'))


#Overall means
for column in data[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]:
    mean_column = data[column].mean()
    std_column = data[column].std()
    print(column, 'MEAN = ', mean_column, '; ' 'STANDARD DEV = ', std_column)

###Really high standard deviations relative to means, so going to cut out any duplicate Pokemon (i.e. 'Megas')
###And take the upper half of the total_points column and analyze that to see if things look any different
data = data[~data['name'].str.contains('Mega')]
data = data[~data['name'].str.contains('Alolan')]
data = data[~data['name'].str.contains('Galarian')]
print(len(data.index))

#Original row count was 1045, and now down to 958 after removing dupes
desc = [('Mean', data['total_points'].mean()), ('StDev', data['total_points'].std()), ('Max Val', data['total_points'].max()), ('Min Val', data['total_points'].min()), ('Median', data['total_points'].median())]
print(desc)

#Mean is 431.23, high variance (stdev = 118.02), median = 450.0
#Thinking median split is the way to go
med = data['total_points'].median()
df = data[data['total_points'] >= med]

desc_uppertier = [('Mean', df['total_points'].mean()), ('StDev', df['total_points'].std()), ('Max Val', df['total_points'].max()), ('Min Val', df['total_points'].min()), ('Median', df['total_points'].median())]
print('ORIGINAL = ', desc, 'UPPER TIER = ', desc_uppertier)
#Cut standard deviation in total_points in half, so this seemed worthwile

###Want to see if there are clusters of different "classes" of Pokemon based on stats (i.e. 'tanks' v. 'attackers' v. 'mixed')
###However, don't want to runinto clusters of 'good' v. 'bad' v. 'mediocre'
    #Going to calculate ratio of each stat to total_points in order to control for just generally stronger Pokemon

#Examining ratios
for column in df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]:
    columnSeriesObj = df[column]
    df['column_rat'] = columnSeriesObj / df['total_points']
    print(column, 'ratio', 'MEAN =', df['column_rat'].mean(), 
    'STANDARD DEV =', df['column_rat'].std())

#Creating columns for ratios
df['hp_rat'] = df['hp'] / df['total_points']
df['attack_rat'] = df['attack'] / df['total_points']
df['defense_rat'] = df['defense'] / df['total_points']
df['sp_atk_rat'] = df['sp_attack'] / df['total_points']
df['sp_def_rat'] = df['sp_defense'] / df['total_points']
df['speed_rat'] = df['speed'] / df['total_points']

###Using K-Means to create clusters based on ratio scores
from sklearn.cluster import KMeans
x = df.iloc[:, -6:].values

wcss = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
print(wcss)

#Determine optimal number of clusters 
'''plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()'''

#Looks like 4 clusters is likely best (tried 6 but saw little variance between clusters)
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
kmeans.fit(x)
y_kmeans = kmeans.fit_predict(x)
df['cluster_km'] = y_kmeans

#Anovas
for column in df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']]:
    columnSeriesObj = df[column]
    model = ols('columnSeriesObj.values ~ cluster_km', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('ANOVA', column, anova_table)

#Compare means
print('MEAN', pd.pivot_table(df, index=df['cluster_km'], 
    values=df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']], 
    aggfunc='mean'))

#Compare standard deviations
print('STANDARD DEV', pd.pivot_table(df, index=df['cluster_km'], 
    values=df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']], 
    aggfunc='std'))

#MANOVA across ratios from k-means clusters
maov_km = MANOVA.from_formula('hp_rat + attack_rat + defense_rat + sp_atk_rat + sp_def_rat + speed_rat ~ cluster_km', data=df)
print(maov_km.mv_test())

###Hierarchical clustering - didn't love results from k-means
'''dendrogram = sch.dendrogram(sch.linkage(x, method='ward', ))
plt.title('Dendrogram')
plt.xlabel('Pokemon (Rows)')
plt.ylabel('Euclidean Distances (b/w rows)')
plt.show()'''

#I like 6-clusters here based on dendrogram, but didn't yield any real differences between groups
#Will use 4 to compare clusters against k-means
from sklearn.cluster import AgglomerativeClustering
y_hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')

y_hc = y_hc.fit_predict(x)

df['cluster_hc'] = y_hc

#ANOVAs
for column in df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']]:
    columnSeriesObj = df[column]
    model = ols('columnSeriesObj.values ~ cluster_hc', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('ANOVA', column, anova_table)

#Compare means
print('MEAN', pd.pivot_table(df, index=df['cluster_hc'], 
    values=df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']], 
    aggfunc='mean'))

#Compare standard deviations
print('STANDARD DEV', pd.pivot_table(df, index=df['cluster_hc'], 
    values=df[['hp_rat', 'attack_rat', 'defense_rat', 'sp_atk_rat', 'sp_def_rat', 'speed_rat']], 
    aggfunc='std'))

#MANOVA across ratios for hierarchical clusters
maov_hc = MANOVA.from_formula('hp_rat + attack_rat + defense_rat + sp_atk_rat + sp_def_rat + speed_rat ~ cluster_km', data=df)
print(maov_hc.mv_test())

###CONCLUSIONS###
#Reduced variance overall by separating upper tier Pokemon from lower tier using median-split method
#Then used unsupervised learning to sort Pokemon into 4 distinct clusters
#Significant differences found between clusters using both K-means and hierarchical clustering methods

###Examining K-means clusters first
#Cluster 1 has highest mean attack ratio (.228) and moderately high defense (.187) and hp (.171)
    #While, special attack and speed are both poor, is generally decent across the board
#Cluster 2 appear to typify the prototypical 'Tank' class - have highest defense (.275) and special defense (.224)
    #This may offset their poor overall health (.128)
    # Perform especially poorly in speed though (.089)
#Cluster 3 fit the mold of a blitz attacker - strong scores in attack (.180), special attack (.185), and speed (.215)
    #Have second worst mean health however (.142) behind Cluster 2
#Cluster 4 are the specialist class -- special attack (.199) and special defense (.184) are both high
    #rangre from middling to poor in other categories however

###Examining clusters derived from hierarchical clustering
#Cluster 1 - shows highest mean attack ratio (.200) and highest HP (.191)
    #However speed is relatively low (at least relative to other stats - .137), so may be a 'heavy hitter'
#Cluster 2 - second highest attack (.195) and far and away highest defense (.251)
    #Coupled with second highest special defense (.174), makes for a great 'tank' class (though speed is very low)
#Cluster 3 - though attack is relatively low (.174), the high special attack (.182) and high speed (.213) make for a good blitz
    #Very low defense (.135) may pose a significant weakness
#Cluster 4 - similar to K-means cluster 4 - is a specialist class; highest special attack (.215) and special defense (.193)
    #Other stats aren't necessarily low, so a decent group overall

###Future directions
#1) Using a total 'against' score, could likely see if Pokemon could be sorted into groups based on types
    #Could see if there are certain Pokemon that have fewer weaknesses relative to others and incorporate that into team building
    #Conversely, are there some that may be significantly effective against large swaths of Pokemon
#2) By converting absolute values for each stat into ordinal ranks, could then sum across stats and sort descending to see which is highest in most stats?
    #However, could be confounded by higher total_points (hence why the ratio was calculated here)