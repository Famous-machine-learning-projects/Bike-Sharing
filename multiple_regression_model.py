import code
"""
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

import Regression_model

df = pd.read_csv('day.csv', encoding='cp1252')

# take a look at the dataset
df.head()
# summarize the data
df.describe()

cdf = df[['season', 'weathersit','temp', 'atemp', 'hum', 'windspeed', 'cnt']]
cdf.head(9)
viz = cdf[['season', 'weathersit','temp', 'atemp', 'hum', 'windspeed', 'cnt']]
viz.hist()
plt.show()

#log transformation:
log_inf=np.log(df['cnt'])
df['log_inf']=log_inf

cdf = df[['season', 'weathersit','temp', 'atemp', 'hum', 'windspeed', 'cnt', 'log_inf']]
cdf.head(9)

plt.scatter(cdf.temp, cdf.cnt,  color='blue')
plt.xlabel("temp")
plt.ylabel("cnt")
plt.show()

plt.scatter(cdf.temp, cdf.log_info,  color='red')
plt.xlabel("temp")
plt.ylabel("log_info")
plt.show()

plt.scatter(cdf.atemp, cdf.cnt,  color='blue')
plt.xlabel("atemp")
plt.ylabel("cnt")
plt.show()

plt.scatter(cdf.atemp, cdf.log_info,  color='red')
plt.xlabel("atemp")
plt.ylabel("log_info")
plt.show()

plt.scatter(cdf.hum, cdf.cnt,  color='blue')
plt.xlabel("hum")
plt.ylabel("cnt")
plt.show()

plt.scatter(cdf.hum, cdf.log_info,  color='red')
plt.xlabel("hum")
plt.ylabel("log_info")
plt.show()


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#print(msk)
#print(~msk)
#print(train)
#print(test)

#temperture:
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(test.temp, test.log_inf,  color='red')
ax1.scatter(train.temp, train.log_inf,  color='blue')
plt.xlabel("temp")
plt.ylabel("answer")
plt.show()

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['temp']])
train_y = np.asanyarray(train[['log_inf']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

Coefficients:  [[1.94455693]]
Intercept:  [7.32038657]


plt.scatter(train.temp, train.log_inf,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("temp")
plt.ylabel("answer")

from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['temp']])
test_y = np.asanyarray(test[['log_inf']])
test_y_ = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.4f" % r2_score(test_y , test_y_) )
per = r2_score(test_y , test_y_)*100
print("R2-score percent: %.2f" % per)

Mean absolute error: 0.34
Residual sum of squares (MSE): 0.19
R2-score: 0.3814
R2-score percent: 38.14



#A temperture:
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(test.atemp, test.log_inf,  color='red')
ax1.scatter(train.atemp, train.log_inf,  color='blue')
plt.xlabel("atemp")
plt.ylabel("answer")
plt.show()

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['atemp']])
train_y = np.asanyarray(train[['log_inf']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

Coefficients:  [[2.20298942]]
Intercept:  [7.23951191]


plt.scatter(train.atemp, train.log_inf,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("atemp")
plt.ylabel("answer")

from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['atemp']])
test_y = np.asanyarray(test[['log_inf']])
test_y_ = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.4f" % r2_score(test_y , test_y_) )
per = r2_score(test_y , test_y_)*100
print("R2-score percent: %.2f" % per)

Mean absolute error: 0.34
Residual sum of squares (MSE): 0.18
R2-score: 0.3912
R2-score percent: 39.12
"""
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['temp', 'atemp']])
y = np.asanyarray(train[['log_inf']])


regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

"""
Coefficients:  [[-0.01859716  2.2236595 ]]
Intercept:  [7.23892938]
"""

y_hat= regr.predict(test[['temp', 'atemp']])
x = np.asanyarray(test[['temp', 'atemp']])
y = np.asanyarray(test[['log_inf']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

"""
Residual sum of squares: 0.18
Variance score: 0.39
"""
