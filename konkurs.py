import pandas as pd
import numpy as np
import gc
np.random.seed(0)

import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

import sys
import re

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from scipy.stats.mstats import winsorize
import xgbfir

train = pd.read_hdf('../input/property.train.h5')
test = pd.read_hdf('../input/property.test.h5') #zbiór testowy bez odpowiedzi

train['price_log'] = np.log(train['price'])

df_all = pd.concat([train, test], sort=False)

col_to_factorize = [column for column in df_all.columns if ':' in column]
for column in col_to_factorize:
    df_all['{}_cat'.format(column)] = df_all[column].factorize()[0]

train = df_all[~df_all['price'].isna()]
test = df_all[df_all['price'].isna()]

feats = train.select_dtypes(include=['number']).columns
black_list = ['price', 'price_log', 'id']
feats = [f for f in feats if f not in black_list]

X_train, X_test, y_train, y_test = train_test_split(train[feats], train['price_log'], test_size=0.3, random_state=0)

model = DummyRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred = np.exp(y_pred)
y_pred[y_pred < 0] = np.min(np.exp(y_test))

mean_absolute_error(y_test, y_pred)

X_test = test[feats]
y_test = test['price_log']
y_pred = model.predict(X_test)
y_pred = np.exp(y_pred)
y_pred[y_pred < 0] = np.min(np.exp(y_test))
test['price'] = y_pred
test[ ['id', 'price'] ].to_csv('../output/simple_model.csv', index=False) 