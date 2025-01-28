# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:18:24 2025

@author: leekai
"""

"""for RF and XGBoost GridSearch, 5 independent vars"""

# Data Preprocessing
# Importing the libraries 
import numpy as np 
# np contains mathematical tools
import pandas as pd
# pd best for importing and managing dataset
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Importing the dataset
dataset = pd.read_csv('SPR_23ALL.csv')
    

# creating a matrix of independent variables
X = dataset.iloc[:,:-1].values
# : means all lines/columns, -1 indicates all columns except 
# the last one
# press f5 to run file and set folder directory

# Creating the dependent variable vector
y = dataset.iloc[:,-1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
# sklearn is scikitlearn, contains libraries to make ML models
# Imputer is for missing data
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
imputer = imputer.fit(X[:,0:6])
# 6 bcs upper bound is not included, indexes in python start at 0
X[:,0:6] = imputer.transform(X[:,0:6])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Hyperparameter Tuning with GridSearchCV

# Define the parameter grid
from sklearn.model_selection import GridSearchCV
rf_param_grid = {
    'n_estimators': [100, 200, 300],   # Number of trees
    'max_depth': [10, 20, None],      # Maximum depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4]     # Minimum samples in a leaf node
}

# Initialize GridSearchCV for Random Forest
rf_grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=0),
    param_grid=rf_param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1  # Use all processors
)

# Fit GridSearchCV to the training data for Random Forest
rf_grid_search.fit(X_train, y_train)

# Best parameters for Random Forest
rf_best_params = rf_grid_search.best_params_
print("Best Parameters for Random Forest:", rf_best_params)

# XGBoost Parameter Grid
xgb_param_grid = {
    'n_estimators': [50, 100, 200],    # Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.2], # Step size shrinkage
    'max_depth': [3, 5, 7],            # Maximum depth of trees
    'subsample': [0.8, 1.0],           # Fraction of data used for each tree
    'colsample_bytree': [0.8, 1.0]     # Fraction of features used per tree
}

# Initialize GridSearchCV for XGBoost
xgb_grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=0, objective='reg:squarederror'),
    param_grid=xgb_param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1  # Use all processors
)

# Fit GridSearchCV to the training data for XGBoost
xgb_grid_search.fit(X_train, y_train)

# Best parameters for XGBoost
xgb_best_params = xgb_grid_search.best_params_
print("Best Parameters for XGBoost:", xgb_best_params)
