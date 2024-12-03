# AR example
from statsmodels.tsa.ar_model import AutoReg
from random import random
import pandas as pd
import numpy as np

# contrived dataset
df = pd.read_csv('C:/Users/Avina/OneDrive/Documents/GitHub/Power-Delhi/megaweadata.csv')
datalen=len(df)
data=df['DELHI'][:5000]
print(data)
# fit model
model = AutoReg(data, lags=1)
model_fit = model.fit()
# make prediction
yhat = model_fit.forecast(steps=(datalen-5000))
print(yhat)
print(df['DELHI'][5000:])

def mean_absolute_percentage_error(y_true, y_pred):
  """
  Calculates mean absolute percentage error (MAPE) between
  two lists of numbers.
  
  Parameters
  ----------
  y_true : array-like of shape = (n_samples)
    Ground truth (correct) target values.
  y_pred : array-like of shape = (n_samples)
    Estimated target values.
  
  Returns
  -------
  mape : float
    Mean absolute percentage error between y_true and y_pred.
  """
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mape = mean_absolute_percentage_error(df['DELHI'][5000:], yhat)
accrate = 100 - mape
print('MAPE: %.3f' % accrate)
