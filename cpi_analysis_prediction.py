
# PROJECT: CPI DATA ANALYSIS AND PREDICTION

# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


# Load the dataset
data = pd.read_csv("C:/Users/spars/Downloads/cpialrl_5.csv")

data = data.drop(columns=['inflation_al', 'inflation_rl'])

# Removing inconsistent data
data = data[data['index_al'] > 200].copy()


# DATA SUMMARY
print("\n========== DATASET INFORMATION ==========")
data.info()

print("\n========== STATISTICAL SUMMARY ==========")
print(data.describe())

print("\n========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== LAST 5 ROWS ==========")
print(data.tail())

print("\n========== DATASET SHAPE ==========")
print(data.shape)

print("\n========== COLUMN NAMES ==========")
print(data.columns)


# CHECKING MISSING VALUES
print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())


# HANDLING MISSING VALUES
data.loc[data["index_al"] == 0, "index_al"] = np.nan
data.loc[data["index_rl"] == 0, "index_rl"] = np.nan

data['index_al'] = data['index_al'].fillna(data['index_al'].mean())
data['index_rl'] = data['index_rl'].fillna(data['index_rl'].mean())

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(data.isnull().sum())


# CHECKING DUPLICATE ROWS
print("Duplicate rows:", data.duplicated().sum())


# RELATIONSHIP BETWEEN RURAL CPI AND AGRICULTURAL CPI
plt.figure(figsize=(8, 4))
sns.scatterplot(x='index_rl', y='index_al', data=data, alpha=0.9)
plt.title('Scatter Plot of Rural CPI vs Agricultural CPI')
plt.xlabel('Rural CPI (index_rl)')
plt.ylabel('Agricultural CPI (index_al)')
plt.show()


# TOP 5 STATES WITH HIGHEST CPI
top_states = (
    data.groupby('state')['index_al']
    .max()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
)

plt.figure(figsize=(8, 4))
sns.barplot(
    x='state',
    y='index_al',
    data=top_states,
    errorbar=None,
    hue='state',
    legend=False,
    palette='viridis'
)
plt.title('Top 5 States with Highest CPI')
plt.xlabel('State')
plt.ylabel('Max CPI')
plt.ylim(1400, 1600)
plt.show()


# DISTRIBUTION OF AGRICULTURAL CPI VALUES
plt.figure(figsize=(8, 4))
plt.hist(
    data['index_al'],
    bins=20,
    color='blue',
    edgecolor='black',
    alpha=0.7
)
plt.title('Histogram of Agricultural CPI')
plt.xlabel('CPI Values')
plt.ylabel('Frequency')
plt.show()


# YEAR-WISE TREND OF AGRICULTURAL CPI
plt.figure(figsize=(8, 4))
year_trend = data.groupby('year')['index_al'].mean()

plt.plot(year_trend.index, year_trend.values, marker='o')
plt.title('Year-wise Trend of Agricultural CPI')
plt.xlabel('Year')
plt.ylabel('Average CPI')
plt.xticks(rotation=45)
plt.show()


# MOVING AVERAGE TREND OF AGRICULTURAL CPI
plt.figure(figsize=(8, 4))
year_trend = data.groupby('year')['index_al'].mean()
moving_avg = year_trend.rolling(3).mean()

plt.plot(year_trend.index, year_trend.values, label='Actual')
plt.plot(year_trend.index, moving_avg.values, label='Moving Average')

plt.title('Moving Average of Agricultural CPI')
plt.xlabel('Year')
plt.ylabel('CPI')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# MINIMUM VS MAXIMUM CPI BY YEAR
plt.figure(figsize=(8, 4))
year_min = data.groupby('year')['index_al'].min()
year_max = data.groupby('year')['index_al'].max()

plt.plot(year_min.index, year_min.values, label='Minimum CPI')
plt.plot(year_max.index, year_max.values, label='Maximum CPI')

plt.title('Minimum vs Maximum CPI by Year')
plt.xlabel('Year')
plt.ylabel('CPI')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# TOP VS BOTTOM STATE CPI USING MEAN VALUE
plt.figure(figsize=(6, 4))
state_avg = data.groupby('state')['index_al'].mean()

top_state = state_avg.idxmax()
bottom_state = state_avg.idxmin()

values = [state_avg.max(), state_avg.min()]
labels = [top_state, bottom_state]

plt.bar(labels, values, color=['green', 'red'])
plt.title('Top vs Bottom State CPI')
plt.xlabel('State')
plt.ylabel('CPI')
plt.show()


# PREPARING DATA FOR ALL MODELS
X = data[['index_rl']]
y = data['index_al']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# LINEAR REGRESSION

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

x_train_scaled = x_scaler.fit_transform(X_train)
x_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(
    y_train.to_frame()
).ravel()

model = LinearRegression()
model.fit(x_train_scaled, y_train_scaled)

# Prediction for normalized Rural CPI
input_value = 0.5

check_value = pd.DataFrame({
    'index_rl': x_scaler.inverse_transform([[input_value]])[0]
})

check_value_scaled = x_scaler.transform(check_value)

result_scaled = model.predict(check_value_scaled)
result = y_scaler.inverse_transform(
    result_scaled.reshape(-1, 1)
).ravel()

print("\n========== LINEAR REGRESSION ==========")
print("Normalized Rural CPI input:", input_value)
print("Predicted Agricultural CPI:", round(result[0], 2))

# Test predictions
y_pred_scaled = model.predict(x_test_scaled)
lr_pred = y_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).ravel()

print("First 5 test predictions:", lr_pred[:5])


# DECISION TREE REGRESSION

dt_model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=10
)

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Prediction using the same input
dt_result = dt_model.predict(check_value)

print("\n========== DECISION TREE REGRESSION ==========")
print("Normalized Rural CPI input:", input_value)
print("Predicted Agricultural CPI:", round(dt_result[0], 2))
print("First 5 test predictions:", dt_pred[:5])

# Decision Tree plot
plt.figure(figsize=(20, 8))

plot_tree(
    dt_model,
    feature_names=['index_rl'],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Regression - CPI Prediction")
plt.show()


# KNN REGRESSION

knn_scaler = MinMaxScaler()

X_train_knn = knn_scaler.fit_transform(X_train)
X_test_knn = knn_scaler.transform(X_test)

knn_model = KNeighborsRegressor(n_neighbors=21)
knn_model.fit(X_train_knn, y_train)

knn_pred = knn_model.predict(X_test_knn)

# Prediction using the same input
check_value_scaled = knn_scaler.transform(check_value)
knn_result = knn_model.predict(check_value_scaled)

print("\n========== KNN REGRESSION ==========")
print("Normalized Rural CPI input:", input_value)
print("Predicted Agricultural CPI:", round(knn_result[0], 2))
print("First 5 test predictions:", knn_pred[:5])


# SUPPORT VECTOR REGRESSION

svr_scaler = MinMaxScaler()

X_train_svr = svr_scaler.fit_transform(X_train)
X_test_svr = svr_scaler.transform(X_test)

# Changed to linear kernel
svr_model = SVR(kernel="linear", C=1000)
svr_model.fit(X_train_svr, y_train)

svr_pred = svr_model.predict(X_test_svr)

# Prediction using the same input
check_value_scaled = svr_scaler.transform(check_value)
svr_result = svr_model.predict(check_value_scaled)

print("\n========== SUPPORT VECTOR REGRESSION ==========")
print("Normalized Rural CPI input:", input_value)
print("Predicted Agricultural CPI:", round(svr_result[0], 2))
print("First 5 test predictions:", svr_pred[:5])

# SVR actual vs predicted plot
plt.figure(figsize=(8, 5))

plt.scatter(
    X_test['index_rl'],
    y_test,
    label='Actual CPI'
)

plt.scatter(
    X_test['index_rl'],
    svr_pred,
    label='Predicted CPI'
)

plt.title("SVR: Actual vs Predicted CPI")
plt.xlabel("Rural CPI")
plt.ylabel("Agricultural CPI")
plt.legend()
plt.show()


# FINAL COMPARISON OF ALL FOUR MODELS

comparison = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Decision Tree Regression',
        'KNN Regression',
        'Support Vector Regression'
    ],
    'MAE': [
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, dt_pred),
        mean_absolute_error(y_test, knn_pred),
        mean_absolute_error(y_test, svr_pred)
    ],
    'MSE': [
        mean_squared_error(y_test, lr_pred),
        mean_squared_error(y_test, dt_pred),
        mean_squared_error(y_test, knn_pred),
        mean_squared_error(y_test, svr_pred)
    ],
    'R2 Score': [
        r2_score(y_test, lr_pred),
        r2_score(y_test, dt_pred),
        r2_score(y_test, knn_pred),
        r2_score(y_test, svr_pred)
    ]
})

print("\n========== FINAL MODEL COMPARISON ==========")
print(comparison.round(4).to_string(index=False))

print("\nBest model according to R2 Score:")
print(comparison.loc[comparison['R2 Score'].idxmax(), 'Model'])
