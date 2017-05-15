from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as pt

# Obtain data
data = pd.read_fwf('brain_body.txt')
x_vals = data[['Body']]
y_vals = data[['Brain']]

# Train a linear regression model
train_model = linear_model.LinearRegression()
train_model.fit(x_vals, y_vals)

# Graph data
pt.scatter(x_vals, y_vals)
pt.plot(x_vals, train_model.predict(x_vals))
pt.show()
