#!/usr/bin/env python
# coding: utf-8

# In[7]:


from __future__ import print_function

#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)
import seaborn as sns
from patsy import dmatrices
import os
sns.set_style('whitegrid')

# Step-2. Import data
os.chdir('/Users/pauline/Documents/Python')
df = pd.read_csv("Tab-Morph.csv")
df = df.dropna()
nsample = 25
#x = np.linspace(0, 25, nsample)
x = df.slope_angle
X = np.column_stack((x, (x - 5)**2))
X = sm.add_constant(X)
beta = [5., 0.5, -0.01]
sig = 0.5
w = np.ones(nsample)
w[nsample * 6//10:] = 3
y_true = np.dot(X, beta)
e = np.random.normal(size=nsample)
y = y_true + sig * w * e 
X = X[:,[0,1]]

# Step-3.
mod_wls = sm.WLS(y, X, weights=1./(w ** 2))
res_wls = mod_wls.fit()
print(res_wls.summary())

# Step-4.
res_ols = sm.OLS(y, X).fit()
print(res_ols.params)
print(res_wls.params)

# Step-5.
se = np.vstack([[res_wls.bse], [res_ols.bse], [res_ols.HC0_se], 
                [res_ols.HC1_se], [res_ols.HC2_se], [res_ols.HC3_se]])
se = np.round(se,4)
colnames = ['x1', 'const']
rownames = ['WLS', 'OLS', 'OLS_HC0', 'OLS_HC1', 'OLS_HC3', 'OLS_HC3']
tabl = SimpleTable(se, colnames, rownames, txt_fmt=default_txt_fmt)
print(tabl)

# Step-6.
covb = res_ols.cov_params()
prediction_var = res_ols.mse_resid + (X * np.dot(covb,X.T).T).sum(1)
prediction_std = np.sqrt(prediction_var)
tppf = stats.t.ppf(0.975, res_ols.df_resid)

# Step-7.
prstd_ols, iv_l_ols, iv_u_ols = wls_prediction_std(res_ols)

# Step-8.
prstd, iv_l, iv_u = wls_prediction_std(res_wls)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, 'o', label="Bathymetric \nObservations", linewidth=.7, c='#0095d9')
ax.plot(x, y_true, '-', c='#1e50a2', label="True", linewidth=.9)
# OLS
ax.plot(x, res_ols.fittedvalues, 'r--', linewidth=.7)
ax.plot(x, iv_u_ols, 'r--', label="Ordinary Least Squares", linewidth=.7)
ax.plot(x, iv_l_ols, 'r--', linewidth=.7)
# WLS
ax.plot(x, res_wls.fittedvalues, '--.', c='#65318e', linewidth=.7, )
ax.plot(x, iv_u, '--', c='#65318e', label="Weighted Least Squares", linewidth=.7)
ax.plot(x, iv_l, '--', c='#65318e', linewidth=.7)
ax.legend(loc="best");
ax.set_xlabel('Slope angle, degree', fontsize=10)

plt.title("Weighted Least Squares of \nslope angle degrees across Mariana Trench by bathymetric profiles", 
          fontsize=12)
plt.annotate('F', xy=(-0.01, 1.06), xycoords="axes fraction", fontsize=18,
           bbox=dict(boxstyle='round, pad=0.3', fc='w', edgecolor='grey', linewidth=1, alpha=0.9))
plt.show()


# In[ ]:




