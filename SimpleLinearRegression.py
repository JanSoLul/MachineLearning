import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

data = pd.read_csv("./linear1.csv")
y = data['GPA']
x1 = data['SAT']
print(x1)
plt.title("Check")
plt.scatter(x1, y)
yhat=0.2750+0.0017*x1 #yHat=0.275+0.0017x1 Regression Line
fig=plt.plot(x1, yhat, lw=4, c='orange', label='Regression Line')
plt.xlabel('SAT', fontsize='20')
plt.ylabel('GPA', fontsize='20')
plt.show()

'''
# y=b0 + b1x1
x = sm.add_constant(x1)

results = sm.OLS(y, x).fit()
#Contain Ordinary Least Square Regression
results.summary()
'''
