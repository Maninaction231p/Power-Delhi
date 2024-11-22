# AR example
from statsmodels.tsa.ar_model import AutoReg
from random import random
# contrived dataset
data = [round(x + random(),2) for x in range(1, 100)]
print(data)
# fit model
model = AutoReg(data, lags=1)
model_fit = model.fit()
# make prediction
yhat = model_fit.forecast(steps=3)
print(yhat)