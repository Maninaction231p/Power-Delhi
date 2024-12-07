import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib


def sarimax_model_train(data,column_name):
    # Ensure 'TIMESLOT' is in datetime format and sort
    data['TIMESLOT'] = pd.to_datetime(data['TIMESLOT'])
    data.sort_values(by='TIMESLOT', inplace=True)
    data.set_index('TIMESLOT', inplace=True)
    print(data)
    # Create lag features (AR terms) to simulate autoregressive component
    lags = 24  # Create lag features for the past 24 hours
    for lag in range(1, lags + 1):
        data[f'lag_{lag}'] = data[column_name].shift(lag)

    # Seasonal features: Hour of the day, Day of the week
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek

    # Exogenous variables (weather data and holidays)
    exog_features = ['temp', 'feelslike', 'dew', 'humidity', 'windspeed', 'isHoliday']

    # Drop rows with missing values due to lag creation
    data.dropna(inplace=True)

    # Prepare feature matrix X and target variable y
    X = data[['hour', 'day_of_week'] + [f'lag_{i}' for i in range(1, lags + 1)] + exog_features]
    y = data[column_name]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)

    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize the features
        ('model', RandomForestRegressor(random_state=0))  # Random Forest Regressor
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
    joblib.dump(pipeline, 'model/sarimaxforestmodel.pkl')
    print("Model saved as 'sarimaxforestmodel.pkl'")

    # Create a DataFrame for predictions and ground truth
    pred = {
        "test": y_test,
        "pred": y_pred
    }
    predd = pd.DataFrame(pred)
    predd.sort_values(by='TIMESLOT', inplace=True)  # Ascending order
    print(predd)
    # Plot the predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.plot( predd['test'], label="Actual", color='blue')
    plt.plot( predd['pred'], label="Predicted", color='orange')
    plt.ylabel(column_name)
    plt.title("Actual vs Predicted Power Usage")
    plt.legend()
    plt.show()

    # Save predictions to a CSV file
    predd.to_csv(f'C:/Users/Avina/OneDrive/Documents/GitHub/Power-Delhi/Code/Code/csv/{column_name}pred.csv', index=False)



# Load the dataset
data = pd.read_csv('C:/Users/Avina/OneDrive/Documents/GitHub/Power-Delhi/megaweadata.csv')
sarimax_model_train(data,'Other')