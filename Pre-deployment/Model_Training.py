# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:09:13 2025

@author: ASUS_ROG

Details:
    includes 5 independent variables
    Models Random Forest, SVR, XGBoost, and Stacked Ensemble Model
"""

# Data Preprocessing
# Importing the libraries 
import numpy as np 
# np contains mathematical tools
import matplotlib.pyplot as plt
# plt helps plot charts
import pandas as pd
# pd best for importing and managing dataset
from sklearn.ensemble import StackingRegressor

import dataframe_image as dfi
 
dataset = pd.read_csv('final_dataset.csv')

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
# interchange median with mean to see which is best for the dataset
imputer = imputer.fit(X[:,0:5])
# 6 bcs upper bound is not included, indexes in python start at 0
X[:,0:5] = imputer.transform(X[:,0:5])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel() 

# Train SVR
from sklearn.svm import SVR
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

# Predicting results for the test set
svr_pred_scaled = svr.predict(X_test)  # Predictions on scaled test set
svr_pred = sc_y.inverse_transform(svr_pred_scaled.reshape(-1, 1)).ravel()  # Inverse transform

# Train XGBoost
from xgboost import XGBRegressor
xgb = XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=7, 
    random_state=0)
xgb.fit(X_train, y_train)

# Predict with XGBoost
xgb_pred_scaled = xgb.predict(X_test)
xgb_pred = sc_y.inverse_transform(xgb_pred_scaled.reshape(-1, 1)).ravel()

# Train Random Forest model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(
    n_estimators= 300,
    max_depth= 20,
    min_samples_split= 5,
    min_samples_leaf= 1,
    random_state=0
)
regressor.fit(X_train, y_train)

# Predict with Random Forest
rf_pred_scaled = regressor.predict(X_test)

# Inverse transform the scaled predictions and actual values
rf_pred = sc_y.inverse_transform(rf_pred_scaled.reshape(-1, 1)).ravel()

# Stacking setup
from sklearn.linear_model import Ridge
base_learners = [
    ('rf', regressor),
    ('svr', svr),
    ('xgb', xgb)
]
meta_learner = Ridge(alpha=1.0)

stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_learner,
    passthrough=True,
    cv=5,
    n_jobs=-1
)

stacking_model.fit(X_train, y_train)

stack_pred_scaled = stacking_model.predict(X_test)
stack_pred = sc_y.inverse_transform(stack_pred_scaled.reshape(-1, 1)).ravel()



# Evaluating the models
from sklearn.metrics import mean_squared_error, r2_score
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"Random Forest - MSE: {rf_mse:.2f}, R²: {rf_r2:.2f}")
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"XGBoost - MSE: {xgb_mse:.2f}, R²: {xgb_r2:.2f}")
svr_mse = mean_squared_error(y_test, svr_pred)
svr_r2 = r2_score(y_test, svr_pred)
print(f"Support Vector Regression - MSE: {svr_mse:.2f}, R²: {svr_r2:.2f}")
stk_mse = mean_squared_error(y_test, stack_pred)
stk_r2 = r2_score(y_test, stack_pred)
print(f"Stacked - MSE: {stk_mse:.2f}, R²: {stk_r2:.2f}")

# Comparing Actual vs Predicted values
comparison = pd.DataFrame(
    {
     'Actual': y_test, 
     'RF Predicted': rf_pred, 
     'XGB Predicted': xgb_pred,
     'SVR Predicted': svr_pred,
     'STK Predicted': stack_pred,
     })
print(comparison)

# Feature Importance
features = [
    'pH', 'Water Temperature', 'Turbidity', 'AP','TDS']  # Update based on dataset
feature_importances = regressor.feature_importances_

# Visualizing Feature Importance
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Dissolved Oxygen Prediction')
plt.show()

# Avoid division by very small Actual values
small_value_threshold = 0.1
adjusted_actual = np.where(comparison['Actual'] < small_value_threshold, small_value_threshold, comparison['Actual'])

# Calculate Percentage Error
comparison['RF Percentage Error'] = abs(comparison['Actual'] - comparison['RF Predicted']) / adjusted_actual * 100
comparison['XGB Percentage Error'] = abs(comparison['Actual'] - comparison['XGB Predicted']) / adjusted_actual * 100
comparison['SVR Percentage Error'] = abs(comparison['Actual'] - comparison['SVR Predicted']) / adjusted_actual * 100
comparison['Stacked Predicted'] = stack_pred
comparison['Stacked % Error'] = abs(comparison['Actual'] - comparison['Stacked Predicted']) / adjusted_actual * 100
comparison['Stacked % Accuracy'] = 100 - comparison['Stacked % Error']

# Calculate Overall Percentage Accuracy
rfa = comparison['RF Percentage Accuracy'] = 100 - comparison['RF Percentage Error']
xgba = comparison['XGB Percentage Accuracy'] = 100 - comparison['XGB Percentage Error']
svra = comparison['SVR Percentage Accuracy'] = 100 - comparison['SVR Percentage Error']
stka = comparison['Stacked % Accuracy'] = 100-comparison['Stacked % Error']

# Remove outliers based on IQR
Q1 = comparison['Actual'].quantile(0.25)
Q3 = comparison['Actual'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
comparison = comparison[(comparison['Actual'] >= lower_bound) & (comparison['Actual'] <= upper_bound)]

# Avoid division by very small Actual values
small_value_threshold = 0.1
adjusted_actual = np.where(comparison['Actual'] < small_value_threshold, small_value_threshold, comparison['Actual'])

# Recalculate Overall Accuracy
overall_accuracy_rf = comparison['RF Percentage Accuracy'].mean()
overall_accuracy_xgb = comparison['XGB Percentage Accuracy'].mean()
overall_accuracy_svr = comparison['SVR Percentage Accuracy'].mean()
print(f"RF Overall Average Percentage Accuracy: {overall_accuracy_rf:.2f}%")
print(f"XGB Overall Average Percentage Accuracy: {overall_accuracy_xgb:.2f}%")
print(f"SVR Overall Average Percentage Accuracy: {overall_accuracy_svr:.2f}%")
overall_accuracy_stack = comparison['Stacked % Accuracy'].mean()
print(f"Stacked Model Overall Average Percentage Accuracy: {overall_accuracy_stack:.2f}%")

# Display Results
print("\nComparison Table with Adjusted Percentage Error and Accuracy:")
print(comparison)

comparison.to_excel('test_set_results.xlsx', index=False)
df = pd.DataFrame(comparison)
dfi.export(df, 'dataframe.png')

# MAPE
mape_rf = np.mean(
    abs(comparison['Actual'] - comparison['RF Predicted']) / adjusted_actual) * 100
mape_xgb = np.mean(
    abs(comparison['Actual'] - comparison['XGB Predicted']) / adjusted_actual) * 100
mape_svr = np.mean(
    abs(comparison['Actual'] - comparison['SVR Predicted']) / adjusted_actual) * 100
mape_stk = np.mean(
    abs(comparison['Actual'] - comparison['Stacked Predicted']) / adjusted_actual) * 100
print(f"RF Mean Absolute Percentage Error (MAPE): {mape_rf:.2f}%")
print(f"XGB Mean Absolute Percentage Error (MAPE): {mape_xgb:.2f}%")
print(f"SVR Mean Absolute Percentage Error (MAPE): {mape_svr:.2f}%")
print(f"STK Mean Absolute Percentage Error (MAPE): {mape_stk:.2f}%")


# Combine predictions for comparison
comparison = pd.DataFrame({
    'Actual': y_test,
    'RF_Predicted': rf_pred,
    'SVR_Predicted': svr_pred,
    'XGB_Predicted': xgb_pred,
    'STK_Predicted': stack_pred,
    'RF %Accuracy': rfa,
    'SVR %Accuracy': svra,
    'XGB %Accuracy': xgba,
    'STK %Accuracy': stka,
})
print("\nComparison Table:")
print(comparison)


# Actual vs Predicted scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test)), y_test, color='maroon', label='Actual', s=10, alpha=0.8)
plt.scatter(range(len(y_test)), rf_pred, color='blue', label='Random Forest Predicted', s=10, alpha=0.3)
plt.scatter(range(len(y_test)), xgb_pred, color='green', label='XGBoost Predicted', s=10, alpha=0.3)
plt.scatter(range(len(y_test)), svr_pred, color='hotpink', label='SVR Predicted', s=10, alpha=0.3)
plt.scatter(range(len(y_test)), stack_pred, color='orange', label='Stacked Predicted', s=10, alpha=0.3)
plt.title('Actual vs Estimated Dissolved Oxygen Levels')
plt.xlabel('Sample Index')
plt.ylabel('Dissolved Oxygen (mg/L)')
plt.legend()
plt.show()


# For saving models
import joblib

# Save the Random Forest model
# joblib.dump(regressor, "random_forest_model.pkl")
joblib.dump(regressor, "random_forest_model_final.pkl")

# Save the SVR model
# joblib.dump(svr, "svr_model.pkl")
joblib.dump(svr, "svr_model_final.pkl")

# Save the XGBoost model
#joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(xgb, "xgb_model_final.pkl")

# joblib.dump(stacking_model, "stacking_model.pkl")
joblib.dump(stacking_model, "stacking_model_final.pkl")

# Save the scalers
# joblib.dump(sc_X, "scaler_X.pkl")
# joblib.dump(sc_y, "scaler_y.pkl")

joblib.dump(sc_X, "scaler_X_final.pkl")
joblib.dump(sc_y, "scaler_y_final.pkl")

