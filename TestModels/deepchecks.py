import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('C:/Users/Avina/OneDrive/Documents/GitHub/Power-Delhi/megaweadata.csv')

# Convert the 'Date' column to datetime and set it as the index
df['TIMESLOT'] = pd.to_datetime(df['TIMESLOT'])
df.set_index('TIMESLOT', inplace=True)


#print(df.shape)
#print(df.info())
duplicate_rows_data = df[df.duplicated()]
#print("number of duplicate rows: ", duplicate_rows_data.shape)

df.drop_duplicates(inplace=True)


# missing value query
df.dropna(inplace=True)


from sklearn.model_selection import train_test_split

# Select relevant features and the target variable (DELHI)
features = df[["temp", "feelslike", "dew", "humidity", "windspeed", "isHoliday"]]
target = df["DELHI"]

print(df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Install PyCaret if not available (uncomment if running locally)
# !pip install pycaret



# Create a dictionary of models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Initialize a scaler
scaler = StandardScaler()

# Evaluate each model
results = {}
for name, model in models.items():
    pipeline = Pipeline([('scaler', scaler), ('model', model)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    results[name] = {"RMSE": rmse, "R2": r2}

print(results)

# Import necessary libraries
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Create a pipeline with scaling and Gradient Boosting
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Gradient Boosting Model Performance:")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Optional: Save the model
import joblib
joblib.dump(pipeline, 'gradient_boosting_model.pkl')
print("Model saved as 'gradient_boosting_model.pkl'")

pred = {
  "test": y_test,
  "pred": y_pred
}
predd = pd.DataFrame(pred)
predd.sort_values(by='TIMESLOT', inplace=True)  # Ascending order
print(predd)

plt.plot(predd['test'])
plt.plot(predd['pred'])
plt.show()


print("1-----------------------------------------------------------------------------------------------------------------------------------------------------------------")

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('C:/Users/Avina/OneDrive/Documents/GitHub/Power-Delhi/megaweadata.csv')

# Ensure 'TIMESLOT' is in datetime format and sort
data['TIMESLOT'] = pd.to_datetime(data['TIMESLOT'])
data.sort_values(by='TIMESLOT', inplace=True)

# Create time-based features
data['time_index'] = (data['TIMESLOT'] - data['TIMESLOT'].min()).dt.total_seconds() / (60 * 60 * 24)  # Days since start
data['sin_day'] = np.sin(2 * np.pi * data['time_index'] / 1)  # Daily seasonality
data['cos_day'] = np.cos(2 * np.pi * data['time_index'] / 1)
data['sin_week'] = np.sin(2 * np.pi * data['time_index'] / 7)  # Weekly seasonality
data['cos_week'] = np.cos(2 * np.pi * data['time_index'] / 7)

# Exogenous variables
exog_features = ['temp', 'humidity', 'isHoliday']

# Prepare feature matrix and target variable
X = data[['time_index', 'sin_day', 'cos_day', 'sin_week', 'cos_week'] + exog_features]
y = data['DELHI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train a Gradient Boosting Regressor (you can replace this with other sklearn models)
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(data['TIMESLOT'][-len(y_test):], y_test, label="Actual", color='blue')
plt.plot(data['TIMESLOT'][-len(y_test):], y_pred, label="Predicted", color='orange')
plt.xlabel("Date")
plt.ylabel("DELHI")
plt.title("Actual vs Predicted Power Usage")
plt.legend()
plt.show()

# Save predictions
predictions = pd.DataFrame({'TIMESLOT': data['TIMESLOT'][-len(y_test):], 'Actual': y_test, 'Predicted': y_pred})
predictions.to_csv('C:/Users/Avina/OneDrive/Documents/GitHub/Power-Delhi/sklearn_forecast.csv', index=False)
print("Predictions saved as 'sklearn_forecast.csv'")




print("2--------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load the dataset
data = pd.read_csv('C:/Users/Avina/OneDrive/Documents/GitHub/Power-Delhi/megaweadata.csv')

# Ensure 'TIMESLOT' is in datetime format and sort
data['TIMESLOT'] = pd.to_datetime(data['TIMESLOT'])
data.sort_values(by='TIMESLOT', inplace=True)
data.set_index('TIMESLOT', inplace=True)

# Create lag features (AR terms) to simulate autoregressive component
lags = 24  # Create lag features for the past 24 hours
for lag in range(1, lags + 1):
    data[f'lag_{lag}'] = data['DELHI'].shift(lag)

# Seasonal features: Hour of the day, Day of the week
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek

# Exogenous variables (weather data and holidays)
exog_features = ['temp', 'feelslike', 'dew', 'humidity', 'windspeed', 'isHoliday']

# Drop rows with missing values due to lag creation
data.dropna(inplace=True)

# Prepare feature matrix X and target variable y
X = data[['hour', 'day_of_week'] + [f'lag_{i}' for i in range(1, lags + 1)] + exog_features]
y = data['DELHI']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)

# Create the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('model', RandomForestRegressor(random_state=42))  # Random Forest Regressor
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Save the model pipeline
joblib.dump(pipeline, 'sarimaxforestmodel.pkl')
print("Model saved as 'sarimaxforestmodel.pkl'")

# Create a DataFrame for predictions and ground truth
pred = {
    "test": y_test,
    "pred": y_pred
}
predd = pd.DataFrame(pred)
predd.sort_values(by='TIMESLOT', inplace=True)  # Ascending order

# Plot the predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot( predd['test'], label="Actual", color='blue')
plt.plot( predd['pred'], label="Predicted", color='orange')
plt.ylabel("DELHI")
plt.title("Actual vs Predicted Power Usage")
plt.legend()
plt.show()

# Save predictions to a CSV file
predd.to_csv('C:/Users/Avina/OneDrive/Documents/GitHub/Power-Delhi/pred.csv', index=False)




print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------")


data = pd.read_csv('C:/Users/Avina/OneDrive/Documents/GitHub/Power-Delhi/megaweadata.csv')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Assuming your data is loaded into 'data'
# Ensure 'TIMESLOT' is in datetime format and sort
data['TIMESLOT'] = pd.to_datetime(data['TIMESLOT'])
data.sort_values(by='TIMESLOT', inplace=True)
data.set_index('TIMESLOT', inplace=True)

# Create lag features (AR terms) to simulate autoregressive component
lags = 24  # Create lag features for the past 24 hours
for lag in range(1, lags + 1):
    data[f'lag_{lag}'] = data['DELHI'].shift(lag)

# Seasonal features: Hour of the day, Day of the week
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek

# Exogenous variables (weather data and holidays)
exog_features = ['temp', 'feelslike', 'dew', 'humidity', 'windspeed', 'isHoliday']

# Drop rows with missing values due to the lag creation
data.dropna(inplace=True)

# Prepare feature matrix X and target variable y
X = data[['hour', 'day_of_week'] + [f'lag_{i}' for i in range(1, lags + 1)] + exog_features]
y = data['DELHI']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForestRegressor (or you can choose another regression model)
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")


pred = {
  "test": y_test,
  "pred": y_pred
}
predd = pd.DataFrame(pred)
predd.sort_values(by='TIMESLOT', inplace=True)  # Ascending order

print(predd)

plt.plot(predd['test'])
plt.plot(predd['pred'])
predd.to_csv('C:/Users/Avina/OneDrive/Documents/GitHub/Power-Delhi/pred.csv')
plt.show()




