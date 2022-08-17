import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

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
