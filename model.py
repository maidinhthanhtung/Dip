from __future__ import print_function
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
sns.set_context("notebook")
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             accuracy_score, roc_curve,
                             precision_recall_curve, f1_score)
from sklearn import metrics
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv("loans.csv")
df = pd.get_dummies(df, columns=["purpose"], drop_first=True)

X = df.loc[:, df.columns != "not.fully.paid"].values
y = df.loc[:, df.columns == "not.fully.paid"].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)

#xu ly missing data
imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp = imp.fit(X_train)
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)

#chuan hoa du lieu
std = RobustScaler()
std.fit(X_train)
X_train = std.transform(X_train)
X_test = std.transform(X_test)

#xy ly data imbalanced
random_oversampler = RandomOverSampler(sampling_strategy=1, random_state=0)
X_res, y_res = random_oversampler.fit_resample(X_train, y_train)

#models
#LR
lr_model = LogisticRegression(penalty="l2",
                                C=0.01, class_weight={0: 0.4,
                                                      1: 0.6})
def logicstic_regression():
  lr_model.fit(X_res, y_res)
  y_pred_proba = lr_model.predict_proba(X_test)[::,1]
  fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
  auc = metrics.roc_auc_score(y_test, y_pred_proba)
  plt.plot(fpr,tpr,label="LR, auc="+str(auc))
  plt.legend(loc=4)
  #plt.show()

logicstic_regression()
pickle.dump(lr_model, open('lr_model.pkl','wb'))