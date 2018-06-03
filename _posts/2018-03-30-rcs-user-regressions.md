---
layout: post
title:  "Looking into Couples' Account Sharing with Python"
date: 2018-03-30 00:00:00 -0400
updated: 2018-05-14 00:00:00 -0400
categories: [blog]
published: false
---

In this post are codes I wrote for the exploratory analysis of the data for a publication that I'm currently working on. The original data was collected from Amazon Mechanical Turk with this [survey](http://cmu.ca1.qualtrics.com/jfe/form/SV_beZL6a2GYEOjgwt).

I intended to study the behavior of sharing online accounts among couples in romantic relationships. To do so, I collected data on the variety of digital accounts people share with their romantic partners, along with data on their age, gender, income, education, relationship status, relationship duration, and so forth. With collected data, I conducted hypothesis tests and regressions to find patterns in people's sharing behavior. 

The publication based on analyses in this post will be available upcoming June in the [Symposium on Usable Privacy and Security (SOUPS)](https://www.usenix.org/conference/soups2018).

## Data Preparation

### 1. Import modules


```python
# import modules
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from IPython.display import Image
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from statsmodels.iolib.summary2 import summary_col
from linearmodels.iv import IV2SLS

import os, string, math, pydot, time, itertools, copy
import numpy as np
import pandas as pd
import seaborn as sb
import statsmodels
import statsmodels.api as sm
import matplotlib.pyplot as plt

# set palette color
sb.set_palette('colorblind', color_codes=True)
# set display option
pd.options.display.max_rows = 4000
```

    /Users/cheul/science/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


### 2. Helper functions


```python
# some helper functions

# find outliers and drop them
def dropOutlier(variable, data):
    t = 1.5*stats.iqr(data[variable])
    Q1 = data[variable].quantile(0.25)
    Q3 = data[variable].quantile(0.75)
    return data[(data[variable] >= Q1-t) & (data[variable] <= Q3+t)]

def dropAndUnion(variables, data):
    prev = None
    res = None
    if len(variables) == 1:
        var = variables[0]
        return dropOutlier(var, data)
    else:
        for var in variables:
            if prev is None:
                prev = dropOutlier(var, data)
            else:
                res = prev.merge(dropOutlier(var, data), how='inner')
        return res

# show distributions of predictors
def showDist(data, variables):
    n = len(variables)
    
    # figure for distribution plots
    plt.figure(figsize=(n*3,3))

    # plot subplots showing the distribution of each variable
    for i in range(n):
        variable = variables[i]
        plt.subplot(1,n,i+1)
        sb.distplot(data[variable])
        sb.despine(offset=10, trim=True)

    plt.tight_layout()

    # figure for boxplots
    plt.figure(figsize=(n*3,3))
    
    for i in range(n):
        variable = variables[i]
        plt.subplot(1,n,i+1)
        sb.boxplot(data[variable])
        sb.despine(offset=10, trim=True)

    plt.tight_layout()

# drop outliers until there isn't any
def dropUntil(data, variables):
    prev = data
    drop_count = 0
    while prev.shape != dropAndUnion(variables, data).shape:
        drop_count += 1
        data = dropAndUnion(variables, data)
        prev = data
    print('dropped outliers for %d iterations'%drop_count)
    return prev

# standardization function
def standardize(data, variable):
    mean = np.mean(data[variable])
    stdev = np.std(data[variable])
    data['std_%s'%variable] = (data[variable]-mean)/stdev
```

### 3. Data


```python
# read file and prepare user df
users = pd.read_csv('users.csv')
drops = ['Unnamed: 0', 'mTurkCode', 'surveyDuration',
         'sexualOrientation', 'partnerGender', 'selfEthnicity', 'partnerEthnicity',
         'securityDiscussion', 'emergencyDiscussion', 'numAccs1', 'numAccs2',
         'numAccs3', 'numAccs4', 'numAccs5']
users = users.drop(drops, axis=1)
```


```python
print(users.shape)
users.head()
```

    (195, 11)





<div style="overflow-x: auto; margin-bottom: 15px;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>relStatus</th>
      <th>relDuration</th>
      <th>cohabStatus</th>
      <th>cohabDuration</th>
      <th>selfAge</th>
      <th>partnerAge</th>
      <th>selfGender</th>
      <th>income</th>
      <th>education</th>
      <th>numShared</th>
      <th>numAccsAll</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>120</td>
      <td>1</td>
      <td>60</td>
      <td>29</td>
      <td>29</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>54</td>
      <td>1</td>
      <td>32</td>
      <td>34</td>
      <td>33</td>
      <td>2</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>38</td>
      <td>1</td>
      <td>2</td>
      <td>22</td>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>26</td>
      <td>1</td>
      <td>16</td>
      <td>29</td>
      <td>28</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>32</td>
      <td>35</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 4. Prepare a dataframe for users


```python
# get the ratio of sharing
users['shareRatio'] = users['numShared']/users['numAccsAll']
# make a new predictor that combines relationship and cohabitation duration
# lets give more 'weight' to cohabitation duration
users['combinedDuration'] = users['relDuration'] - users['cohabDuration'] + users['cohabDuration']*1.5
# add age difference as a new predictor
users['ageDifference'] = users['selfAge'] - users['partnerAge']
# drop rows where the ratio of sharing equal NaN (=0/0)
users = users.dropna()
# drop users who weren't dating at the time of participation
users = users[users['relStatus'] != 3]
# drop users of unconventional gender
users = users[users['selfGender'] != 5]
# rename columns
users = users.rename({'relStatus':'married', 'cohabStatus':'cohabiting', 'selfGender':'female'}, axis='columns')
# recode values
users['married'] = np.where(users['married'] == 1, 0, 1)
users['cohabiting'] = np.where(users['cohabiting'] == 1, 1, 0)
users['female'] = np.where(users['female'] == 1, 1, 0)

# standardize some predictors
# for var in ['combinedDuration', 'relDuration', 'cohabDuration', 'selfAge', 'income', 'education', 'shareRatio']:
#     standardize(users, var)

# make a copy of original dataframe
users_copy = copy.deepcopy(users)
```


```python
print(users.shape)
users.head()
```

    (182, 14)





<div style="overflow-x: auto; margin-bottom: 15px;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>married</th>
      <th>relDuration</th>
      <th>cohabiting</th>
      <th>cohabDuration</th>
      <th>selfAge</th>
      <th>partnerAge</th>
      <th>female</th>
      <th>income</th>
      <th>education</th>
      <th>numShared</th>
      <th>numAccsAll</th>
      <th>shareRatio</th>
      <th>combinedDuration</th>
      <th>ageDifference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>120</td>
      <td>1</td>
      <td>60</td>
      <td>29</td>
      <td>29</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>12</td>
      <td>0.416667</td>
      <td>150.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>54</td>
      <td>1</td>
      <td>32</td>
      <td>34</td>
      <td>33</td>
      <td>0</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>30</td>
      <td>0.200000</td>
      <td>70.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>38</td>
      <td>1</td>
      <td>2</td>
      <td>22</td>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>18</td>
      <td>0.111111</td>
      <td>39.0</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>26</td>
      <td>1</td>
      <td>16</td>
      <td>29</td>
      <td>28</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>0.250000</td>
      <td>34.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>35</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>6.0</td>
      <td>-3</td>
    </tr>
  </tbody>
</table>
</div>



### 4. Drop outliers from the data


```python
# distributions of predictors before dropping any outliers
showDist(users_copy, ['combinedDuration', 'selfAge', 'shareRatio', 'ageDifference'])
```


![png]({{'/assets/img/rcs-user-regressions/output_13_0.png'}})



![png]({{'/assets/img/rcs-user-regressions/output_13_1.png'}})



```python
users = users_copy

# drop outliers as much as possible
users = dropUntil(users, ['selfAge'])

# distribution of predictors after dropping outliers
showDist(users, ['combinedDuration', 'selfAge', 'shareRatio', 'ageDifference'])
```

    dropped outliers for 2 iterations



![png]({{'/assets/img/rcs-user-regressions/output_14_1.png'}})



![png]({{'/assets/img/rcs-user-regressions/output_14_2.png'}})



```python
print(users.shape)
users.describe()
```

    (174, 14)





<div style="overflow-x: auto; margin-bottom: 15px;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>married</th>
      <th>relDuration</th>
      <th>cohabiting</th>
      <th>cohabDuration</th>
      <th>selfAge</th>
      <th>partnerAge</th>
      <th>female</th>
      <th>income</th>
      <th>education</th>
      <th>numShared</th>
      <th>numAccsAll</th>
      <th>shareRatio</th>
      <th>combinedDuration</th>
      <th>ageDifference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>174.000000</td>
      <td>174.000000</td>
      <td>174.000000</td>
      <td>174.000000</td>
      <td>174.000000</td>
      <td>174.000000</td>
      <td>174.000000</td>
      <td>174.000000</td>
      <td>174.000000</td>
      <td>174.000000</td>
      <td>174.000000</td>
      <td>174.000000</td>
      <td>174.000000</td>
      <td>174.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.339080</td>
      <td>78.551724</td>
      <td>0.747126</td>
      <td>60.413793</td>
      <td>33.264368</td>
      <td>32.701149</td>
      <td>0.431034</td>
      <td>2.908046</td>
      <td>3.977011</td>
      <td>5.448276</td>
      <td>17.856322</td>
      <td>0.358119</td>
      <td>108.758621</td>
      <td>0.563218</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.474763</td>
      <td>77.752945</td>
      <td>0.435914</td>
      <td>78.781699</td>
      <td>7.216647</td>
      <td>9.121676</td>
      <td>0.496650</td>
      <td>1.339763</td>
      <td>1.253684</td>
      <td>5.476716</td>
      <td>10.841317</td>
      <td>0.301911</td>
      <td>116.214294</td>
      <td>5.721274</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>19.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.500000</td>
      <td>-20.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>28.000000</td>
      <td>26.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>11.000000</td>
      <td>0.111111</td>
      <td>29.250000</td>
      <td>-2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>50.500000</td>
      <td>1.000000</td>
      <td>27.000000</td>
      <td>32.000000</td>
      <td>31.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>16.000000</td>
      <td>0.258333</td>
      <td>66.750000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>117.750000</td>
      <td>1.000000</td>
      <td>81.000000</td>
      <td>38.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>8.000000</td>
      <td>24.750000</td>
      <td>0.520833</td>
      <td>146.250000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>346.000000</td>
      <td>1.000000</td>
      <td>360.000000</td>
      <td>53.000000</td>
      <td>58.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>39.000000</td>
      <td>59.000000</td>
      <td>1.000000</td>
      <td>517.500000</td>
      <td>32.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# make variables for median breakdown
cols = {'relDuration':'relAboveMdn', 'cohabDuration':'cohAboveMdn',
        'combinedDuration':'comAboveMdn', 'selfAge':'ageAboveMdn',
        'income':'incAboveMdn', 'education':'eduAboveMdn', 'shareRatio':'ratioAboveMdn'}
for col in cols:
    new_col = cols[col]
    users[new_col] = np.where(users[col] > np.median(users[col]), 1, 0)

# make dummy variables for income and education
# users = pd.get_dummies(users, columns=['income', 'education'],
#                        prefix={'income':'inc', 'education':'edu'}, drop_first=True)
```


```python
np.median(users['shareRatio'])
```




    0.2583333333333333



## Regression Analysis

### 1. Exploratory visualizations with single predictors


```python
# exploratory analysis with single predictors
plt.figure(figsize=(12,4))

plt.subplot(131)
sb.regplot(x='combinedDuration', y='shareRatio', data=users, color='g')
sb.despine(offset=10, trim=True)

plt.subplot(132)
sb.regplot(x='selfAge', y='shareRatio', data=users, color='b')
sb.despine(offset=10, trim=True)

plt.subplot(133)
sb.regplot(x='selfAge', y='combinedDuration', data=users, color='r')
sb.despine(offset=10, trim=True)

plt.tight_layout()
plt.show()
```


![png]({{'/assets/img/rcs-user-regressions/output_20_0.png'}})



```python
plt.figure(figsize=(12,4))

plt.subplot(131)
sb.regplot(x='ageDifference', y='shareRatio', data=users)
sb.despine(offset=10, trim=True)

plt.subplot(132)
sb.regplot(x='ageDifference', y='combinedDuration', data=users)
sb.despine(offset=10, trim=True)

plt.subplot(133)
sb.regplot(x='selfAge', y='ageDifference', data=users)
sb.despine(offset=10, trim=True)

plt.tight_layout()
plt.show()
```


![png]({{'/assets/img/rcs-user-regressions/output_21_0.png'}})



```python
plt.figure(figsize=(12,4))

plt.subplot(131)
sb.regplot(x='relDuration', y='ratioAboveMdn', data=users, logistic=True)
sb.despine(offset=10, trim=True)

plt.subplot(132)
sb.regplot(x='cohabDuration', y='ratioAboveMdn', data=users, logistic=True)
sb.despine(offset=10, trim=True)

plt.subplot(133)
sb.regplot(x='combinedDuration', y='ratioAboveMdn', data=users, logistic=True)
sb.despine(offset=10, trim=True)

plt.tight_layout()
plt.show()
```


![png]({{'/assets/img/rcs-user-regressions/output_22_0.png'}})


### 2. Compare ratio of sharing across medians of different variables


```python
# make plots to compare distribution of sharing ratio across medians
plt.figure(figsize=(12,3))

plt.subplot(141)
sb.boxplot(x='relAboveMdn', y='shareRatio', data=users)
sb.despine(offset=10, trim=True)

plt.subplot(142)
sb.boxplot(x='cohAboveMdn', y='shareRatio', data=users)
sb.despine(offset=10, trim=True)

plt.subplot(143)
sb.boxplot(x='incAboveMdn', y='shareRatio', data=users)
sb.despine(offset=10, trim=True)

plt.subplot(144)
sb.boxplot(x='eduAboveMdn', y='shareRatio', data=users)
sb.despine(offset=10, trim=True)

plt.tight_layout()

plt.figure(figsize=(12,3))

plt.subplot(141)
sb.boxplot(x='married', y='shareRatio', data=users)
sb.despine(offset=10, trim=True)

plt.subplot(142)
sb.boxplot(x='cohabiting', y='shareRatio', data=users)
sb.despine(offset=10, trim=True)

plt.subplot(143)
sb.boxplot(x='female', y='shareRatio', data=users)
sb.despine(offset=10, trim=True)

plt.subplot(144)
sb.boxplot(x='ageAboveMdn', y='shareRatio', data=users)
sb.despine(offset=10, trim=True)

plt.tight_layout()
```


![png]({{'/assets/img/rcs-user-regressions/output_24_0.png'}})



![png]({{'/assets/img/rcs-user-regressions/output_24_1.png'}})


### 3. Logistic regression with recursive feature elimination
#### [1] Notes on logistic regression,
* First, logistic regression does not require a linear relationship between the dependent and independent variables.
* Second, the error terms (residuals) do not need to be normally distributed.
* Third, homoscedasticity is not required. (in other words, the variance of DV need not stay constant as IV varies)
* Finally, the dependent variable in logistic regression is not measured on an interval or ratio scale.

#### and assumptions of logistic regression
* First, binary logistic regression requires the dependent variable to be binary and ordinal logistic regression requires the dependent variable to be ordinal.
* Second, logistic regression requires the observations to be independent of each other.  In other words, the observations should not come from repeated measurements or matched data.
* Third, logistic regression requires there to be little or no multicollinearity among the independent variables.  This means that the independent variables should not be too highly correlated with each other.
* Fourth, logistic regression assumes linearity of independent variables and log odds.  although this analysis does not require the dependent and independent variables to be related linearly, it requires that the independent variables are linearly related to the log odds.
* Finally, logistic regression typically requires a large sample size.  A general guideline is that you need at minimum of 10 cases with the least frequent outcome for each independent variable in your model. For example, if you have 5 independent variables and the expected probability of your least frequent outcome is .10, then you would need a minimum sample size of 500 (10*5 / .10).

[1]: http://www.statisticssolutions.com/assumptions-of-logistic-regression/


```python
# calculates VIFs for columns in a df = X
def calculate_vif(X):
    variables = X.columns
    vif = {var:variance_inflation_factor(X[variables].values, variables.get_loc(var)) for var in X.columns}
    return vif

# set the number of independent variables to create regression models
num_iv = 7

excludeThese = ['shareRatio', 'ratioAboveMdn', 'numShared',
                'numAccsAll', 'income', 'education',
                'relDuration', 'cohabDuration', 'relAboveMdn', 'cohAboveMdn',
                'comAboveMdn', 'ageAboveMdn', 'partnerAge', 'combinedDuration', 'ageDifference']

dv_log = ['ratioAboveMdn']
iv_log = [i for i in users.columns.tolist() if i not in excludeThese]

dv_lin = ['shareRatio']
iv_lin = [i for i in users.columns.tolist() if i not in excludeThese]

logreg = LogisticRegression()
rfe_log = RFE(logreg, num_iv)
rfe_log = rfe_log.fit(users[iv_log], users[dv_log])

selected_iv_log = [iv_log[i] for i in range(len(iv_log)) if rfe_log.support_[i]]
vifs_log = calculate_vif(users[selected_iv_log])
logit_res = sm.Logit(users[dv_log], users[selected_iv_log]).fit()
print(logit_res.summary())
print(vifs_log)
print(np.exp(logit_res.conf_int()), '\n')
print(np.exp(logit_res.params), '\n')
print(logit_res.pvalues, '\n')

linreg = LinearRegression()
rfe_lin = RFE(linreg, num_iv)
rfe_lin = rfe_lin.fit(users[iv_lin], users[dv_lin])

selected_iv_lin = [iv_lin[i] for i in range(len(iv_lin)) if rfe_lin.support_[i]]
vifs_lin = calculate_vif(users[selected_iv_lin])
lin_res = sm.OLS(users[dv_lin], users[selected_iv_lin]).fit()
print(lin_res.summary())
print(vifs_lin)
```

    Optimization terminated successfully.
             Current function value: 0.614692
             Iterations 5
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:          ratioAboveMdn   No. Observations:                  174
    Model:                          Logit   Df Residuals:                      168
    Method:                           MLE   Df Model:                            5
    Date:                Fri, 30 Mar 2018   Pseudo R-squ.:                  0.1132
    Time:                        14:37:12   Log-Likelihood:                -106.96
    converged:                       True   LL-Null:                       -120.61
                                            LLR p-value:                 4.981e-05
    ===============================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    married         0.8959      0.391      2.293      0.022       0.130       1.662
    cohabiting      1.4660      0.428      3.423      0.001       0.627       2.306
    selfAge        -0.0328      0.012     -2.653      0.008      -0.057      -0.009
    female         -0.4106      0.352     -1.166      0.244      -1.101       0.280
    incAboveMdn     0.2860      0.399      0.717      0.473      -0.496       1.068
    eduAboveMdn    -0.4251      0.338     -1.257      0.209      -1.088       0.238
    ===============================================================================
    {'married': 1.9401025238263816, 'cohabiting': 4.577881786202723, 'selfAge': 5.7584871409404785, 'female': 1.9571313244666926, 'incAboveMdn': 1.596919517942987, 'eduAboveMdn': 2.0194099087558137}
                        0          1
    married      1.138937   5.268047
    cohabiting   1.871126  10.029517
    selfAge      0.944602   0.991479
    female       0.332600   1.322639
    incAboveMdn  0.609231   2.908423
    eduAboveMdn  0.336917   1.268303 
    
    married        2.449485
    cohabiting     4.332030
    selfAge        0.967757
    female         0.663257
    incAboveMdn    1.331128
    eduAboveMdn    0.653691
    dtype: float64 
    
    married        0.021852
    cohabiting     0.000620
    selfAge        0.007989
    female         0.243645
    incAboveMdn    0.473211
    eduAboveMdn    0.208709
    dtype: float64 
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             shareRatio   R-squared:                       0.619
    Model:                            OLS   Adj. R-squared:                  0.606
    Method:                 Least Squares   F-statistic:                     45.51
    Date:                Fri, 30 Mar 2018   Prob (F-statistic):           8.70e-33
    Time:                        14:37:12   Log-Likelihood:                -30.743
    No. Observations:                 174   AIC:                             73.49
    Df Residuals:                     168   BIC:                             92.44
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    married         0.0460      0.053      0.862      0.390      -0.059       0.151
    cohabiting      0.2266      0.055      4.110      0.000       0.118       0.335
    selfAge         0.0067      0.002      4.259      0.000       0.004       0.010
    female         -0.0778      0.047     -1.639      0.103      -0.172       0.016
    incAboveMdn    -0.0436      0.054     -0.805      0.422      -0.151       0.063
    eduAboveMdn    -0.0333      0.046     -0.731      0.466      -0.123       0.057
    ==============================================================================
    Omnibus:                       23.982   Durbin-Watson:                   1.799
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               29.339
    Skew:                           0.978   Prob(JB):                     4.26e-07
    Kurtosis:                       3.469   Cond. No.                         93.7
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    {'married': 1.9401025238263816, 'cohabiting': 4.577881786202723, 'selfAge': 5.7584871409404785, 'female': 1.9571313244666926, 'incAboveMdn': 1.596919517942987, 'eduAboveMdn': 2.0194099087558137}


    /Users/cheul/science/lib/python3.6/site-packages/sklearn/utils/validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /Users/cheul/science/lib/python3.6/site-packages/sklearn/utils/validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /Users/cheul/science/lib/python3.6/site-packages/scipy/linalg/basic.py:1226: RuntimeWarning: internal gelsd driver lwork query error, required iwork dimension not returned. This is likely the result of LAPACK bug 0038, fixed in LAPACK 3.2.2 (released July 21, 2010). Falling back to 'gelss' driver.
      warnings.warn(mesg, RuntimeWarning)


### 4. Hierarchical logistic regression
For a hierarchical logistic regression with sharing ratio above or below the median as a dependent variable, add predictors in the following order
* First w/ relationship characteristics as predictors - marriage, cohabitation
* Then w/ age
* Then w/ gender
* Then w/ demographic variables: income and education

#### 4-1. Prepare IV and DV


```python
# set independent and independent variables
iv1 = ['married']
iv2 = ['cohabiting']
iv3 = iv1 + iv2
iv4 = iv3 + ['selfAge']
iv5 = iv4 + ['female']
iv6 = iv5 + ['incAboveMdn']
iv7 = iv6 + ['eduAboveMdn']
dv = ['ratioAboveMdn']
iv_all = [iv1, iv2, iv3, iv4, iv5, iv6, iv7]

# logit_res1 = sm.Logit(users[dv], users[iv1]).fit()
# print(np.exp(logit_res1.params), '\n')
# print(np.exp(logit_res1.conf_int()), '\n')
# print(logit_res1.pvalues)

# make a skeletal dataframe
res_table = pd.DataFrame(index=range(len(iv_all)), columns=iv_all[-1]+['rsq'])
# iterate over sets of independent variables and fill in cells of the dataframe
for i in range(len(iv_all)):
    iv_set = iv_all[i]
    # fit a logistic model with given IVs
    res = sm.Logit(users[dv], users[iv_set]).fit()
    # get parameters of a model
    params = np.exp(res.params)
    conf_int = np.exp(res.conf_int())
    pvals = res.pvalues
    rsq = res.prsquared
    # fill in cells
    for odd, param_name, pval in zip(params, params.index, pvals):
        cellValue = '%1.2f (%1.2f,%1.2f)'%(odd, conf_int.loc[param_name, 0], conf_int.loc[param_name, 1])
        if pval < 0.05:
            cellValue += '*'
        elif pval < 0.01:
            cellValue += '**'
        elif pval < 0.001:
            cellValue += '***'
        res_table.loc[i, param_name] = cellValue
    res_table.loc[i, 'rsq'] = '%1.3f'%rsq
```

    Optimization terminated successfully.
             Current function value: 0.671190
             Iterations 4
    Optimization terminated successfully.
             Current function value: 0.678103
             Iterations 4
    Optimization terminated successfully.
             Current function value: 0.670058
             Iterations 4
    Optimization terminated successfully.
             Current function value: 0.623173
             Iterations 5
    Optimization terminated successfully.
             Current function value: 0.619721
             Iterations 5
    Optimization terminated successfully.
             Current function value: 0.619276
             Iterations 5
    Optimization terminated successfully.
             Current function value: 0.614692
             Iterations 5



```python
# table summarizing all results
res_table
```




<div style="overflow-x: auto; margin-bottom: 15px;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>married</th>
      <th>cohabiting</th>
      <th>selfAge</th>
      <th>female</th>
      <th>incAboveMdn</th>
      <th>eduAboveMdn</th>
      <th>rsq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.11 (1.22,3.63)*</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.032</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>1.50 (1.06,2.13)*</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.022</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.82 (0.90,3.70)</td>
      <td>1.16 (0.73,1.84)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.033</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.47 (1.18,5.15)*</td>
      <td>4.17 (1.83,9.49)*</td>
      <td>0.96 (0.94,0.98)*</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.62 (1.24,5.54)*</td>
      <td>4.43 (1.92,10.21)*</td>
      <td>0.96 (0.94,0.98)*</td>
      <td>0.68 (0.34,1.36)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.106</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.54 (1.19,5.45)*</td>
      <td>4.36 (1.89,10.09)*</td>
      <td>0.96 (0.94,0.98)*</td>
      <td>0.69 (0.35,1.37)</td>
      <td>1.16 (0.55,2.45)</td>
      <td>NaN</td>
      <td>0.107</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.45 (1.14,5.27)*</td>
      <td>4.33 (1.87,10.03)*</td>
      <td>0.97 (0.94,0.99)*</td>
      <td>0.66 (0.33,1.32)</td>
      <td>1.33 (0.61,2.91)</td>
      <td>0.65 (0.34,1.27)</td>
      <td>0.113</td>
    </tr>
  </tbody>
</table>
</div>




```python
# simple table showing number of participants for different combinations of marriage and cohabitation
table = sm.stats.Table.from_data(users[["married", "cohabiting"]])
table.table
```




    array([[43., 72.],
           [ 1., 58.]])




```python
married_and_cohabiting = users[(users['married'] == 1) & (users['cohabiting'] == 1)]
not_married_and_cohabiting = users[(users['married'] == 0) & (users['cohabiting'] == 0)]
cohabiting_but_not_married = users[(users['married'] == 0) & (users['cohabiting'] == 1)]

print(np.mean(married_and_cohabiting['relDuration']), np.std(married_and_cohabiting['relDuration']))
print(np.mean(cohabiting_but_not_married['relDuration']), np.std(cohabiting_but_not_married['relDuration']))
print(np.mean(not_married_and_cohabiting['relDuration']), np.std(not_married_and_cohabiting['relDuration']))
```

    144.48275862068965 90.34402473978628
    57.888888888888886 42.96334211199239
    23.162790697674417 22.99032175802653



```python
print(np.mean(married_and_cohabiting['shareRatio']), np.std(married_and_cohabiting['shareRatio']))
print(np.mean(cohabiting_but_not_married['shareRatio']), np.std(cohabiting_but_not_married['shareRatio']))
print(np.mean(not_married_and_cohabiting['shareRatio']), np.std(not_married_and_cohabiting['shareRatio']))
```

    0.445826515099993 0.2648081897423066
    0.37925391665420827 0.2835264404652674
    0.21081658248934054 0.3208688329393022



```python
sb.distplot(married_and_cohabiting['shareRatio'])
sb.distplot(cohabiting_but_not_married['shareRatio'])
sb.distplot(not_married_and_cohabiting['shareRatio'])
sb.despine(offset=10, trim=True)
```


![png](/assets/img/rcs-user-regressions/output_34_0.png)


### 5. Linear regression with best subset selection

#### 5-1. Define helper functions


```python
def processSubset(feature_set):
# Fit model on feature_set and calculate RSS
    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    vifs = calculate_vif(X[list(feature_set)])
    return {"model":regr, "RSS":RSS, "VIF":vifs}

def getBest(k):
    tic = time.time()
    results = []
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo))
        
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    
    print("Processed ", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model
```

### 5-2. Build models and select the model with the best subset of features


```python
# best subset selection with different number of variables
models = pd.DataFrame(columns=["RSS", "model", "VIF"])
features = ['married', 'relDuration', 'cohabiting', 'cohabDuration', 'selfAge', 'ageDifference',
            'female', 'relAboveMdn', 'cohAboveMdn', 'ageAboveMdn', 'incAboveMdn', 'eduAboveMdn']
y = users['shareRatio']
X = users[features]

tic = time.time()
for i in range(2, len(features)+1):
    models.loc[i] = getBest(i)

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")
```

    /Users/cheul/science/lib/python3.6/site-packages/ipykernel_launcher.py:19: FutureWarning: 'argmin' is deprecated. Use 'idxmin' instead. The behavior of 'argmin' will be corrected to return the positional minimum in the future. Use 'series.values.argmin' to get the position of the minimum now.


    Processed  66 models on 2 predictors in 0.30315279960632324 seconds.
    Processed  220 models on 3 predictors in 1.2347002029418945 seconds.
    Processed  495 models on 4 predictors in 3.354867935180664 seconds.
    Processed  792 models on 5 predictors in 5.647736072540283 seconds.
    Processed  924 models on 6 predictors in 7.72772479057312 seconds.
    Processed  792 models on 7 predictors in 8.980084896087646 seconds.
    Processed  495 models on 8 predictors in 6.218998908996582 seconds.
    Processed  220 models on 9 predictors in 2.7559492588043213 seconds.
    Processed  66 models on 10 predictors in 0.951589822769165 seconds.
    Processed  12 models on 11 predictors in 0.22101306915283203 seconds.
    Processed  1 models on 12 predictors in 0.022063016891479492 seconds.
    Total elapsed time: 37.451167821884155 seconds.


#### 5-3. Explore results


```python
# put summary of results in a table
num_regr = list(range(2,len(features)+1))
all_models = [models.loc[i, 'model'] for i in num_regr]
info_dict={'R-squared' : lambda x: "{:.2f}".format(x.rsquared),
           'No. observations' : lambda x: "{0:d}".format(int(x.nobs))}

results_table = summary_col(results=all_models,
                            float_format='%0.2f',
                            stars = True,
                            model_names=num_regr,
                            info_dict=info_dict,
                            regressor_order=features)

results_table.add_title('Table 1 - OLS Regressions, Best Subset Selection')

print(results_table)
```

                                    Table 1 - OLS Regressions, Best Subset Selection
    =================================================================================================================
                        2       3        4        5       6        7        8        9        10       11       12   
    -----------------------------------------------------------------------------------------------------------------
    married                           0.08     0.14**  0.12**   0.12**   0.13**   0.13**   0.13**   0.14**   0.14**  
                                      (0.05)   (0.06)  (0.06)   (0.06)   (0.06)   (0.06)   (0.06)   (0.06)   (0.06)  
    relDuration                                        -0.00*** -0.00**                                      -0.00   
                                                       (0.00)   (0.00)                                       (0.00)  
    cohabiting       0.22*** 0.20***  0.18***  0.20*** 0.16***  0.16***  0.18***  0.17***  0.17***  0.17***  0.17*** 
                     (0.05)  (0.05)   (0.05)   (0.06)  (0.06)   (0.06)   (0.06)   (0.06)   (0.06)   (0.06)   (0.06)  
    cohabDuration                              -0.00**                   -0.00*** -0.00*** -0.00*** -0.00**  -0.00   
                                               (0.00)                    (0.00)   (0.00)   (0.00)   (0.00)   (0.00)  
    selfAge          0.01*** 0.01***  0.01***  0.01*** 0.01***  0.01***  0.01***  0.01***  0.01***  0.01***  0.01*** 
                     (0.00)  (0.00)   (0.00)   (0.00)  (0.00)   (0.00)   (0.00)   (0.00)   (0.00)   (0.00)   (0.00)  
    ageDifference                                                                          0.00     0.00     0.00    
                                                                                           (0.00)   (0.00)   (0.00)  
    female                                                      -0.06    -0.07    -0.07    -0.07    -0.07    -0.06   
                                                                (0.05)   (0.05)   (0.05)   (0.05)   (0.05)   (0.05)  
    relAboveMdn                                        0.14**   0.14**   0.11*    0.09     0.08     0.09     0.10    
                                                       (0.06)   (0.06)   (0.06)   (0.08)   (0.08)   (0.08)   (0.09)  
    cohAboveMdn                                                                   0.05     0.06     0.05     0.05    
                                                                                  (0.09)   (0.09)   (0.09)   (0.09)  
    ageAboveMdn              -0.14*** -0.17*** -0.14** -0.17*** -0.17*** -0.17*** -0.17*** -0.17*** -0.17*** -0.18***
                             (0.05)   (0.06)   (0.06)  (0.06)   (0.06)   (0.06)   (0.06)   (0.06)   (0.06)   (0.06)  
    incAboveMdn                                                                                     -0.03    -0.03   
                                                                                                    (0.05)   (0.05)  
    eduAboveMdn                                                          -0.07    -0.06    -0.06    -0.06    -0.06   
                                                                         (0.04)   (0.04)   (0.04)   (0.04)   (0.04)  
    R-squared        0.61    0.62     0.63     0.64    0.65     0.65     0.66     0.66     0.66     0.66     0.66    
    No. observations 174     174      174      174     174      174      174      174      174      174      174     
    =================================================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01



```python
# the best model chosen based on the heuristics of simplicity (low number of variables) and low mean VIF
best_features = users[models.loc[4, 'model'].model.exog_names]
best_model = sm.OLS(y, best_features).fit()
best_vif = calculate_vif(best_features)
print(best_model.summary())
# look at individual VIFs for multicollinearity
print('\n',best_vif)
# and the mean of VIFs
# print('\n',np.mean(list(best_vif.values())))
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             shareRatio   R-squared:                       0.630
    Model:                            OLS   Adj. R-squared:                  0.621
    Method:                 Least Squares   F-statistic:                     72.34
    Date:                Fri, 30 Mar 2018   Prob (F-statistic):           1.10e-35
    Time:                        14:37:50   Log-Likelihood:                -28.241
    No. Observations:                 174   AIC:                             64.48
    Df Residuals:                     170   BIC:                             77.12
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    married         0.0818      0.054      1.523      0.130      -0.024       0.188
    cohabiting      0.1788      0.055      3.276      0.001       0.071       0.286
    selfAge         0.0083      0.002      5.031      0.000       0.005       0.012
    ageAboveMdn    -0.1694      0.056     -3.021      0.003      -0.280      -0.059
    ==============================================================================
    Omnibus:                       19.152   Durbin-Watson:                   1.693
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               22.639
    Skew:                           0.881   Prob(JB):                     1.21e-05
    Kurtosis:                       3.132   Cond. No.                         107.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
     {'married': 2.05594607839861, 'cohabiting': 4.668453561737817, 'selfAge': 6.6815853806192536, 'ageAboveMdn': 3.259320511383283}



```python
# import file for econometrics
%run econtools_metrics.py

# Wu-Hausman Endogeneity test
# takes form: y ~ a + b + c (str), data: dataFrame, variable: variable to be tested for endogeneity
eqn = 'shareRatio ~ combinedDuration + selfAge'
print(wu_test(form=eqn, data=users, variable='selfAge').summary())
```

    WU TEST: The p_value of the added residual is 8.8070e-03
         This is significant at the alpha=0.05 level
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             shareRatio   R-squared:                       0.004
    Model:                            OLS   Adj. R-squared:                 -0.008
    Method:                 Least Squares   F-statistic:                    0.3167
    Date:                Fri, 30 Mar 2018   Prob (F-statistic):              0.729
    Time:                        14:37:51   Log-Likelihood:                -37.686
    No. Observations:                 174   AIC:                             81.37
    Df Residuals:                     171   BIC:                             90.85
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    Intercept            0.0008      0.000      4.999      0.000       0.000       0.001
    combinedDuration    -0.0003      0.000     -1.218      0.225      -0.001       0.000
    selfAge              0.0116      0.001     10.836      0.000       0.010       0.014
    resid_selfAge       -0.0109      0.004     -2.650      0.009      -0.019      -0.003
    ==============================================================================
    Omnibus:                       16.454   Durbin-Watson:                   1.710
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               18.558
    Skew:                           0.776   Prob(JB):                     9.34e-05
    Kurtosis:                       2.609   Cond. No.                     1.38e+18
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 2.38e-30. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.



```python
best_residual = y - best_model.predict()
plt.hist(best_residual)
print(stats.shapiro(best_residual))
# P-P plot of residuals against the normal distribution
ax = plt.figure(figsize=(4,4))
residual_pp = stats.probplot(best_residual, plot=plt)
```

    (0.9278461933135986, 1.2837902829687664e-07)



![png]({{'/assets/img/rcs-user-regressions/output_44_1.png'}})



![png]({{'/assets/img/rcs-user-regressions/output_44_2.png'}})


Is this a sufficiently good model?
* R-squared is quite high
* Not all, but p-values are small
* Coefficient directions seem reasonable; positive coefficients for marriage and cohabitation, negative correlations for age and gender (being female) are noticeable
* Low VIFs -- low multicollinearity

Some problems:
* The non-normal distribution of residuals
* Endogeneity? -- the mean of residuals is small = -3.78864519739971e-05, but is this small enough? (and does the mean equal the expected value of residual/error term given X?)
* There likely is an endogeneity in variables that characterize relationships, ie. relationship duration, cohabiting duration, marriage, and cohabitation; its possible to hypothesize that longer a couple has been together, more likely that they are to share accounts, and more accounts a couple shares, more likely that they are to stay longer
* Could age be an instrument for relationship/cohabitation duration? -- maybe yes, since it's reasonable to assume that older a person is, there's higher chance that he or she have been in a relationship or been cohabiting for a longer period of time, also it seems unlikely that age alone will affect someone's propensity to share an account.

### 6. Linear regression with forward stepwise selection

#### 6-1. Define helper functions


```python
def processSubset(feature_set):
# Fit model on feature_set and calculate RSS
    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def forward(predictors):
    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns if p not in predictors]
    tic = time.time()
    results = []
    for p in remaining_predictors:
        results.append(processSubset(predictors+[p]))
        
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic))
    
    # Return the best model, along with some other useful information about the model
    return best_model
```

#### 6-2. Build models


```python
# do forward selection of predictors
models2 = pd.DataFrame(columns=["RSS", "model"])
features = ['married', 'relDuration', 'cohabiting', 'cohabDuration', 'selfAge',
            'female', 'relAboveMdn', 'cohAboveMdn', 'ageAboveMdn', 'incAboveMdn', 'eduAboveMdn']
y = users['shareRatio']
X = users[features]

tic = time.time()
predictors = []
for i in range(1,len(X.columns)+1):
    models2.loc[i] = forward(predictors)
    predictors = models2.loc[i]["model"].model.exog_names
toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")
```

    Processed  11 models on 1 predictors in 0.022861003875732422
    Processed  10 models on 2 predictors in 0.03093719482421875
    Processed  9 models on 3 predictors in 0.015971899032592773
    Processed  8 models on 4 predictors in 0.0172121524810791
    Processed  7 models on 5 predictors in 0.014009714126586914
    Processed  6 models on 6 predictors in 0.012506961822509766
    Processed  5 models on 7 predictors in 0.008921146392822266
    Processed  4 models on 8 predictors in 0.007986783981323242
    Processed  3 models on 9 predictors in 0.006115913391113281
    Processed  2 models on 10 predictors in 0.005542755126953125
    Processed  1 models on 11 predictors in 0.0032968521118164062
    Total elapsed time: 0.18485593795776367 seconds.


    /Users/cheul/science/lib/python3.6/site-packages/ipykernel_launcher.py:20: FutureWarning: 'argmin' is deprecated. Use 'idxmin' instead. The behavior of 'argmin' will be corrected to return the positional minimum in the future. Use 'series.values.argmin' to get the position of the minimum now.


#### 6-3. Results


```python
# put summary of results in a table
num_regr = list(range(1,len(features)+1))
all_models = [models2.loc[i, 'model'] for i in num_regr]
info_dict={'R-squared' : lambda x: "{:.2f}".format(x.rsquared),
           'No. observations' : lambda x: "{0:d}".format(int(x.nobs))}

results_table = summary_col(results=all_models,
                            float_format='%0.2f',
                            stars = True,
                            model_names=num_regr,
                            info_dict=info_dict,
                            regressor_order=features)

results_table.add_title('Table 2 - OLS Regressions, Forward Stepwise Selection')

print(results_table)
```

                                 Table 2 - OLS Regressions, Forward Stepwise Selection
    ================================================================================================================
                        1       2       3        4        5       6        7        8        9        10       11   
    ----------------------------------------------------------------------------------------------------------------
    married                                   0.08     0.14**  0.12**   0.12**   0.13**   0.13**   0.13**   0.13**  
                                              (0.05)   (0.06)  (0.06)   (0.06)   (0.06)   (0.06)   (0.06)   (0.06)  
    relDuration                                                                                             -0.00   
                                                                                                            (0.00)  
    cohabiting       0.41*** 0.22*** 0.20***  0.18***  0.20*** 0.18***  0.18***  0.18***  0.17***  0.17***  0.17*** 
                     (0.03)  (0.05)  (0.05)   (0.05)   (0.06)  (0.06)   (0.06)   (0.06)   (0.06)   (0.06)   (0.06)  
    cohabDuration                                      -0.00** -0.00*** -0.00*** -0.00*** -0.00*** -0.00**  -0.00   
                                                       (0.00)  (0.00)   (0.00)   (0.00)   (0.00)   (0.00)   (0.00)  
    selfAge                  0.01*** 0.01***  0.01***  0.01*** 0.01***  0.01***  0.01***  0.01***  0.01***  0.01*** 
                             (0.00)  (0.00)   (0.00)   (0.00)  (0.00)   (0.00)   (0.00)   (0.00)   (0.00)   (0.00)  
    female                                                                       -0.07    -0.07    -0.07    -0.07   
                                                                                 (0.05)   (0.05)   (0.05)   (0.05)  
    relAboveMdn                                                0.11*    0.11*    0.11*    0.09     0.09     0.10    
                                                               (0.06)   (0.06)   (0.06)   (0.08)   (0.08)   (0.09)  
    cohAboveMdn                                                                           0.05     0.04     0.04    
                                                                                          (0.09)   (0.09)   (0.09)  
    ageAboveMdn                      -0.14*** -0.17*** -0.14** -0.15*** -0.16*** -0.17*** -0.17*** -0.17*** -0.18***
                                     (0.05)   (0.06)   (0.06)  (0.06)   (0.06)   (0.06)   (0.06)   (0.06)   (0.06)  
    incAboveMdn                                                                                    -0.03    -0.03   
                                                                                                   (0.05)   (0.05)  
    eduAboveMdn                                                         -0.06    -0.07    -0.06    -0.06    -0.06   
                                                                        (0.04)   (0.04)   (0.04)   (0.04)   (0.04)  
    R-squared        0.57    0.61    0.62     0.63     0.64    0.65     0.65     0.66     0.66     0.66     0.66    
    No. observations 174     174     174      174      174     174      174      174      174      174      174     
    ================================================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01


### 7. Linear regression with backward stepwise selection

#### 7-1. Helper functions


```python
def backward(predictors):
    tic = time.time()
    results = []
    for combo in itertools.combinations(predictors, len(predictors)-1):
        results.append(processSubset(combo))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)-1, "predictors in", (toc-tic))
    
    # Return the best model, along with some other useful information about the model
    return best_model
```

#### 7-2. Build models


```python
# do backward selection of predictors
models3 = pd.DataFrame(columns=["RSS", "model"], index = range(1,len(X.columns)))

tic = time.time()
predictors = X.columns

while(len(predictors) > 1):
    models3.loc[len(predictors)-1] = backward(predictors)
    predictors = models3.loc[len(predictors)-1]["model"].model.exog_names

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")
```

    Processed  11 models on 10 predictors in 0.024876117706298828
    Processed  10 models on 9 predictors in 0.021832942962646484
    Processed  9 models on 8 predictors in 0.019124984741210938
    Processed  8 models on 7 predictors in 0.01589798927307129
    Processed  7 models on 6 predictors in 0.021747350692749023
    Processed  6 models on 5 predictors in 0.01593494415283203
    Processed  5 models on 4 predictors in 0.014760017395019531
    Processed  4 models on 3 predictors in 0.008947134017944336
    Processed  3 models on 2 predictors in 0.014951229095458984
    Processed  2 models on 1 predictors in 0.0072078704833984375
    Total elapsed time: 0.17861700057983398 seconds.


    /Users/cheul/science/lib/python3.6/site-packages/ipykernel_launcher.py:11: FutureWarning: 'argmin' is deprecated. Use 'idxmin' instead. The behavior of 'argmin' will be corrected to return the positional minimum in the future. Use 'series.values.argmin' to get the position of the minimum now.
      # This is added back by InteractiveShellApp.init_path()


#### 7-3. Results


```python
# put summary of results in a table
num_regr = list(range(1,len(features)))
all_models = [models3.loc[i, 'model'] for i in num_regr]
info_dict={'R-squared' : lambda x: "{:.2f}".format(x.rsquared),
           'No. observations' : lambda x: "{0:d}".format(int(x.nobs))}

results_table = summary_col(results=all_models,
                            float_format='%0.2f',
                            stars = True,
                            model_names=num_regr,
                            info_dict=info_dict,
                            regressor_order=features)

results_table.add_title('Table 3 - OLS Regressions, Backward Stepwise Selection')

print(results_table)
```

                            Table 3 - OLS Regressions, Backward Stepwise Selection
    =======================================================================================================
                        1       2       3        4        5       6        7        8        9        10   
    -------------------------------------------------------------------------------------------------------
    married                                   0.08     0.14**  0.12**   0.12**   0.13**   0.13**   0.13**  
                                              (0.05)   (0.06)  (0.06)   (0.06)   (0.06)   (0.06)   (0.06)  
    cohabiting       0.41*** 0.22*** 0.20***  0.18***  0.20*** 0.18***  0.18***  0.18***  0.17***  0.17*** 
                     (0.03)  (0.05)  (0.05)   (0.05)   (0.06)  (0.06)   (0.06)   (0.06)   (0.06)   (0.06)  
    cohabDuration                                      -0.00** -0.00*** -0.00*** -0.00*** -0.00*** -0.00** 
                                                       (0.00)  (0.00)   (0.00)   (0.00)   (0.00)   (0.00)  
    selfAge                  0.01*** 0.01***  0.01***  0.01*** 0.01***  0.01***  0.01***  0.01***  0.01*** 
                             (0.00)  (0.00)   (0.00)   (0.00)  (0.00)   (0.00)   (0.00)   (0.00)   (0.00)  
    female                                                                       -0.07    -0.07    -0.07   
                                                                                 (0.05)   (0.05)   (0.05)  
    relAboveMdn                                                0.11*    0.11*    0.11*    0.09     0.09    
                                                               (0.06)   (0.06)   (0.06)   (0.08)   (0.08)  
    cohAboveMdn                                                                           0.05     0.04    
                                                                                          (0.09)   (0.09)  
    ageAboveMdn                      -0.14*** -0.17*** -0.14** -0.15*** -0.16*** -0.17*** -0.17*** -0.17***
                                     (0.05)   (0.06)   (0.06)  (0.06)   (0.06)   (0.06)   (0.06)   (0.06)  
    incAboveMdn                                                                                    -0.03   
                                                                                                   (0.05)  
    eduAboveMdn                                                         -0.06    -0.07    -0.06    -0.06   
                                                                        (0.04)   (0.04)   (0.04)   (0.04)  
    R-squared        0.57    0.61    0.62     0.63     0.64    0.65     0.65     0.66     0.66     0.66    
    No. observations 174     174     174      174      174     174      174      174      174      174     
    =======================================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01


### 8. Two-Stage Least Squares Regression (2SLS)
By using 2SLS, we can deal with an endogeneity in variables


```python
# first select predictors for OLS
predictors = ['relDuration', 'cohabDuration', 'selfAge', 'female']
X = users[predictors]
y = users['shareRatio']

# fit model and get summary
model = sm.OLS(y, X).fit()

# add constant term and fit model
X_c = statsmodels.tools.tools.add_constant(X)
model_c = sm.OLS(y, X_c).fit()

# put summaries of models in a nice table
info_dict={'R-squared' : lambda x: "{:.2f}".format(x.rsquared),
           'No. observations' : lambda x: "{0:d}".format(int(x.nobs))}

results_table = summary_col(results=[model, model_c],
                            float_format='%0.2f',
                            stars = True,
                            model_names=['w/o constant', 'w/ constant'],
                            info_dict=info_dict,
                            regressor_order=predictors+['const'])
results_table.add_title('Table 4 - OLS Regressions, Selected Predictors W/ & W/O a constant')
print(results_table)
```

    Table 4 - OLS Regressions, Selected Predictors W/ & W/O a constant
    =========================================
                     w/o constant w/ constant
    -----------------------------------------
    relDuration      0.00         0.00       
                     (0.00)       (0.00)     
    cohabDuration    -0.00        0.00       
                     (0.00)       (0.00)     
    selfAge          0.01***      0.00       
                     (0.00)       (0.00)     
    female           -0.03        -0.04      
                     (0.05)       (0.05)     
    const                         0.33***    
                                  (0.13)     
    R-squared        0.57         0.01       
    No. observations 174          174        
    =========================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01



```python
# check VIFs for predictors
calculate_vif(X)
```




    {'cohabDuration': 21.287206430230956,
     'female': 2.0183143377281505,
     'relDuration': 28.61588453259341,
     'selfAge': 3.460865445214197}




```python
plt.figure(figsize=(8,4))

plt.subplot(121)
sb.distplot(users['combinedDuration'])
sb.despine(offset=10, trim=True)

plt.subplot(122)
sb.regplot(x='combinedDuration', y='shareRatio', data=users, robust=True)
sb.despine(offset=10, trim=True)

plt.tight_layout()
```


![png]({{'/assets/img/rcs-user-regressions/output_63_0.png'}})



```python
# re-select predictors for OLS
predictors = ['combinedDuration', 'selfAge', 'married']
X = users[predictors]

# fit model and get summary
model = sm.OLS(y, X).fit()

# add constant term and fit model
X_c = statsmodels.tools.tools.add_constant(X)
model_c = sm.OLS(y, X_c).fit()

# make a table summarizing models
results_table = summary_col(results=[model, model_c],
                            float_format='%0.2f',
                            stars = True,
                            model_names=['w/o constant', 'w/ constant'],
                            info_dict=info_dict,
                            regressor_order=['const']+predictors)
print(results_table)
```

    
    =========================================
                     w/o constant w/ constant
    -----------------------------------------
    const                         0.34***    
                                  (0.12)     
    combinedDuration -0.00*       -0.00      
                     (0.00)       (0.00)     
    selfAge          0.01***      -0.00      
                     (0.00)       (0.00)     
    married          0.15**       0.16***    
                     (0.06)       (0.06)     
    R-squared        0.58         0.04       
    No. observations 174          174        
    =========================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01



```python
calculate_vif(X)
```




    {'combinedDuration': 3.368593654823898,
     'married': 2.4285132120669903,
     'selfAge': 2.4319397443094393}




```python
print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             shareRatio   R-squared:                       0.584
    Model:                            OLS   Adj. R-squared:                  0.577
    Method:                 Least Squares   F-statistic:                     80.11
    Date:                Fri, 30 Mar 2018   Prob (F-statistic):           2.05e-32
    Time:                        14:38:29   Log-Likelihood:                -38.355
    No. Observations:                 174   AIC:                             82.71
    Df Residuals:                     171   BIC:                             92.19
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    combinedDuration    -0.0005      0.000     -1.881      0.062      -0.001    2.48e-05
    selfAge              0.0105      0.001      9.963      0.000       0.008       0.013
    married              0.1470      0.062      2.382      0.018       0.025       0.269
    ==============================================================================
    Omnibus:                       15.854   Durbin-Watson:                   1.767
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               18.365
    Skew:                           0.793   Prob(JB):                     0.000103
    Kurtosis:                       2.872   Cond. No.                         431.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
reg_2sls = IV2SLS(dependent=users['shareRatio'],
                  exog=None,
                  endog=users[['combinedDuration']],
                  instruments=users[['selfAge']]).fit()

print(reg_2sls.summary)
```

                              IV-2SLS Estimation Summary                          
    ==============================================================================
    Dep. Variable:             shareRatio   R-squared:                      0.1101
    Estimator:                    IV-2SLS   Adj. R-squared:                 0.1049
    No. Observations:                 174   F-statistic:                    88.725
    Date:                Fri, Mar 30 2018   P-value (F-stat)                0.0000
    Time:                        14:38:29   Distribution:                  chi2(1)
    Cov. Estimator:                robust                                         
                                                                                  
                                    Parameter Estimates                                 
    ====================================================================================
                      Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------------
    combinedDuration     0.0029     0.0003     9.4194     0.0000      0.0023      0.0035
    ====================================================================================
    
    Endogenous: combinedDuration
    Instruments: selfAge
    Robust Covariance (Heteroskedastic)
    Debiased: False


### 9. Decision Trees


```python
# set sharing ratio as target
target = users['shareRatio']
# select featurs for decision tree regression (using median breakdown)
fset = ['married', 'cohabiting', 'relAboveMdn', 'cohAboveMdn', 'selfAge', 'female', 'incAboveMdn', 'eduAboveMdn']
features = users[fset]
# fit decision tree with features
dt_reg = tree.DecisionTreeRegressor()
dt = dt_reg.fit(features, target)
# get dot data from fitted decision tree
dt_dot = tree.export_graphviz(dt, out_file=None,
                              feature_names=fset, filled=True,
                              rounded=True, special_characters=True)
# get decision tree as dot graph
dt_graph = pydot.graph_from_dot_data(dt_dot)
# save decision tree graph
with open('graph.png', 'w') as f:
    dt_graph[0].write_png('graph.png')
# display decision tree
Image(dt_graph[0].create_png())
```




![png]({{'/assets/img/rcs-user-regressions/output_69_0.png'}})




```python
dt.score(features, target)
```




    0.8761167797089106




```python
dt.feature_importances_
```




    array([0.0359595 , 0.09616758, 0.09127467, 0.03346974, 0.5786662 ,
           0.04633269, 0.06833235, 0.04979729])




```python
fset = ['relDuration', 'cohabDuration', 'selfAge']
features = users[fset]
# fit decision tree with features
dt_reg = tree.DecisionTreeRegressor()
dt = dt_reg.fit(features, target)
# get dot data from fitted decision tree
dt_dot = tree.export_graphviz(dt, out_file=None,
                              feature_names=fset, filled=True,
                              rounded=True, special_characters=True)
# get decision tree as dot graph
dt_graph = pydot.graph_from_dot_data(dt_dot)
# save decision tree graph
with open('graph.png', 'w') as f:
    dt_graph[0].write_png('graph.png')
# display decision tree
Image(dt_graph[0].create_png())
```




![png]({{'/assets/img/rcs-user-regressions/output_72_0.png'}})




```python
dt.score(features, target)
```




    0.9993100320259264




```python
dt.feature_importances_
```




    array([0.35486379, 0.33550101, 0.30963519])


