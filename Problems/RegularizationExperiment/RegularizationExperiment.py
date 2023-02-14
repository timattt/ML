import pandas as pd
import numpy as np

#
# Load data
#

url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df_wine = pd.read_csv(url, header=None)
df_wine.columns = ['метка класса', 'алкоголь', 'яблочная кислота', 'зола',
                   'щелочность золы', 'магний', 'всего фенола', 'флаваноиды',
                   'фенолы нефлаваноидные', 'проантоцианины',
                   'интенсивность цвета', 'оттенок', 'разбавление', 'пролин']

print(df_wine.head())

#
# Transform data
#

from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

#
# Test
#

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']

weights = []
params = []

for c in range(-4, 6):
    lr = LogisticRegression(penalty='l1', solver='liblinear', C = 10**c, random_state = 0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
    
weights = np.array(weights)

for column, color in zip(range(len(weights)), colors):
    plt.plot(params, weights[:, column],
             label = df_wine.columns[column+1], color=color)
    
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('весовой коэф')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.legend()
plt.show()