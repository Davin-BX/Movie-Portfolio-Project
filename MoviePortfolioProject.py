#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Import Libraries

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

#Read in the Data

df = pd.read_csv(r'C:\Users\davin\downloads\movies.csv')


# In[62]:


# Let's Look at the Data

df.head()


# In[33]:


# Looking to see if There is any Missing Data

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))


# In[34]:


# Dropping rows with Missing Data Types

df = df.dropna()


# In[36]:


# Identifying Data Types for the Columns

df.dtypes


# In[38]:


# Change Data Type of Columns

df['budget'] = df['budget'].astype('int64')

df['gross'] = df['gross'].astype('int64')


# In[51]:


# Creating a Correct Year Column

df['yearcorrect'] = df['released'].astype(str).str.split().str[2]

df


# In[58]:


# Viewing all Data

pd.set_option('display.max_rows', None)


# In[60]:


# Ordering data by Gross Data Descending

df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[63]:


# Dropping Any Duplicates

df.drop_duplicates()


# In[65]:


# Building a Scatter Plot with Budget vs Gross Revenue

plt.scatter(x=df['budget'], y=df['gross'])

plt.title('Budget vs Gross Earnings')

plt.xlabel('Gross Earnings')

plt.ylabel('budget for Film')

plt.show


# In[67]:


# Plot Budget vs Gross Revenue using Sendborn

sns.regplot(x='budget', y='gross',data=df, scatter_kws={"color": "red"}, line_kws={"color": "blue"})


# In[84]:


# Looking at the Correlation

df.corr(numeric_only=True)


# In[97]:


# Looking at the Correlation Matrix

correlation_matrix = df.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.show()


# In[87]:


# Looking at the Company
df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
df_numerized


# In[95]:


correlation_matrix = df_numerized.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matrix for Numerical Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[98]:


# Comparing the Correlations of the Recorded Data to Identify which data Correlates closer to gross earnings
correlation_mat = df_numerized.corr()

corr_pairs = correlation_mat.unstack()

corr_pairs


# In[140]:


upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=-1).astype(bool))

sorted_pairs = upper_triangle.stack().sort_values(ascending=True)

top_correlated_pairs = sorted_pairs[sorted_pairs >= 0.5]

print("Top correlated pairs:")
print(top_correlated_pairs)

