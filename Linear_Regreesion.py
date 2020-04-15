%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0,10.0)
data = pd.read_csv('headbrain.csv')
print(data.shape)
data.head()


#liner Regression
x= data['Head Size(cm^3)'].values
y = data['Brain Weight(grams)'].values
mean_x= np.mean(x)
mean_y = np.mean(y)
n = len(x)
num = 0
den = 0
for i in range(n):
    num += (x[i] - mean_x) * (y[i] - mean_y)
    den += (x[i] - mean_x)**2
slope = num/den
const = mean_y - (slope * mean_x)
print(slope, const)



max_x = np.mean(x) + 100
min_x = np.mean(y) - 100
eqnx = np.linespace(min_x,min_y,1000)
eqny = (slope * mean_x) + const
plt.show(eqnx,eqny)
plt.scatter(eqnx, eqny)



#checking the goodness of model using r^2 method
num1 = 0
den1 = 0
for i in range(n):
    y_pred = (slope * x[i]) + const
    num1 += (y[i] - mean_y) ** 2
    den1 += (y[i] - y_pred) ** 2
#formulae
r2 = 1 - (num1/den1)
print(r2)


#using skict learn libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
x = x.reshape(m,1)
reg = LinearRegression()
reg = reg.fit(x,y)
y_predict = reg.predict(x)
r2_score = reg.score(x,y)
print(r2_score)
