import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# 
# SKIPPED DATA PROBLEM
#

# READING DATA
raw_data =  '''A,B,C,D
                1.0,2.0,3.0,4.0
                5.0,6.0,,8.0
                10.0,11.0,12.0,'''
                
df = pd.read_csv(StringIO(raw_data))
print(df)

# REMOVING NAN
# removing lines that contains nan
print(df.dropna())

# removing features that contains nan
print(df.dropna(axis=1))

# REPLACING NAN
imr = SimpleImputer(missing_values = pd.NA, strategy='mean')
imr.fit(df.values)
print(imr.transform(df.values))

#
# PARSING CATEGORIZED DATA
#

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                  ['red', 'L', 13.5, 'class2'],
                  ['blue', 'XL', 15.3, 'class3']])
df.columns = ['color', 'size', 'price', 'label']
print(df)

# MAPPING CLASSIFIED DATA INTO NUMBERS
size_mapping = {'M':3, 'L':2, 'XL':1}
df['size'] = df['size'].map(size_mapping)

print(df)

# ONE-HOT encoding

print(OneHotEncoder().fit(df.values).transform(df.values).toarray())
