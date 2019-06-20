#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import os
%matplotlib inline
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)
import seaborn as sns
from patsy import dmatrices

sns.set_style('whitegrid')

os.chdir('/Users/pauline/Documents/Python')
df = pd.read_csv("Tab-Morph.csv")
df = df.dropna()
nsample = 25
x = df.plate_pacif
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

mod_wls = sm.WLS(y, X, weights=1./(w ** 2))
res_wls = mod_wls.fit()
print(res_wls.summary())

res_ols = sm.OLS(y, X).fit()
print(res_ols.params)
print(res_wls.params)

se = np.vstack([[res_wls.bse], [res_ols.bse], [res_ols.HC0_se],
                [res_ols.HC1_se], [res_ols.HC2_se], [res_ols.HC3_se]])
se = np.round(se,4)
colnames = ['x1', 'const']
rownames = ['WLS', 'OLS', 'OLS_HC0', 'OLS_HC1', 'OLS_HC3', 'OLS_HC3']
tabl = SimpleTable(se, colnames, rownames, txt_fmt=default_txt_fmt)
print(tabl)

covb = res_ols.cov_params()
prediction_var = res_ols.mse_resid + (X * np.dot(covb,X.T).T).sum(1)
prediction_std = np.sqrt(prediction_var)
tppf = stats.t.ppf(0.975, res_ols.df_resid)

prstd_ols, iv_l_ols, iv_u_ols = wls_prediction_std(res_ols)

prstd, iv_l, iv_u = wls_prediction_std(res_wls)

bbox_props = dict(boxstyle='round, pad=0.3', fc='w',
                  edgecolor='grey', linewidth=1, alpha=0.9)

fig, ax = plt.subplots(figsize=(8,6), dpi=300)
ax.plot(x, y, 'o', label="Bathymetric \nObservations", linewidth=.7)
ax.plot(x, y_true, 'b-', label="True", linewidth=.9)
# OLS
ax.plot(x, res_ols.fittedvalues, 'r--')
ax.plot(x, iv_u_ols, 'r--', label="Ordinary Least Squares", linewidth=.7)
ax.plot(x, iv_l_ols, 'r--', linewidth=.7)
# WLS
ax.plot(x, res_wls.fittedvalues, 'g--.')
ax.plot(x, iv_u, 'g--', label="Weighted Least Squares", linewidth=.7)
ax.plot(x, iv_l, 'g--', linewidth=.7)
ax.legend(loc="best");
ax.set_xlabel('Number of observations', fontsize=10)

plt.title("Weighted Least Squares of data distribution \
          \nat Pacific Plate by 25 bathymetric profiles",
          fontsize=14)
plt.annotate('A', xy=(-0.01, 1.06), xycoords="axes fraction",
             fontsize=18, bbox=bbox_props)

# visualizing and saving
plt.tight_layout()
plt.subplots_adjust(top=0.87, bottom=0.08,
                    left=0.10, right=0.95,
                    hspace=0.2, wspace=0.2
                    )
plt.savefig('plot_WLSa.png', dpi=300)
plt.show()
