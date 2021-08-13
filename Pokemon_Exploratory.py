##Getting to know the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('pokedex_(Update_04.21).csv')
print(len(data.index))

print(list(data.columns))

print(pd.crosstab(index=data['generation'], columns='count'))
print(pd.crosstab(index=data['status'], columns='count'))
print(pd.crosstab(index=data['species'], columns='count'))
print(pd.crosstab(index=data['type_1'], columns='count'))

#Interested in 'Sub Legendary' from status
subleg = data.loc[data['status'] == 'Sub Legendary']
print(subleg['name'])
print(pd.crosstab(index=subleg['type_1'], columns='count'))

##Do sub legendary and legendary have higher stats on average -- assume yes, but let's see the numbers
#Overall mean
for column in data[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]:
    mean_column = data[column].mean()
    print(column, mean_column)

#Mean by groups
for column in data[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]:
    grouped_df = data.groupby('status')
    mean_grouped_col = grouped_df[column].mean()
    print(column, mean_grouped_col)

#%% ANOVA
#get ANOVA table as R-esque output
import statsmodels.api as sm
from statsmodels.formula.api import ols
#This is to do it once for a single column
model = ols('attack ~ status', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

#This loops through every stat
for column in data[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]:
    columnSeriesObj = data[column]
    model = ols('columnSeriesObj.values ~ status', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(column, anova_table)

#%% MANOVA
from statsmodels.multivariate.manova import MANOVA
maov = MANOVA.from_formula('hp + attack + defense + sp_attack + sp_defense + speed ~ status', data=data)
print(maov.mv_test())

##Lets see what pokemon are the highest in each stat
for column in data[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]:
    col_max = data.loc[data[column].idxmax()]
    print(column, '; ', col_max)

for column in data[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]:
    column_ratio = data[column] / data['total_points']
    print(column, column_ratio.mean())

