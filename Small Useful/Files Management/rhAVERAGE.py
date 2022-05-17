import pandas as pd 
import os

dire= (r'F:\Experiments\Crystal Pendant\FocusedAnalysis\rhS')
names = os.listdir(dire) 
os.chdir(dire)
a, b, c = [], [], []
for name in names:
	df = pd.read_csv(name, skiprows=1, sep='\\t')
	a.append(name)
	b.append(df.iloc[:,3].mean())
	c.append(df.iloc[:,3].std())

# print(df.iloc[:,3].mean())
df2 = pd.DataFrame()
df2['name'] = a
df2['RH'] = b 
df2['std'] = c

print(df2)
df2.to_csv(r'F:\Experiments\Crystal Pendant\FocusedAnalysis\rhS\summup.csv')
