# match date from file creation with name from slice (FIJI)

import pandas as pd

df = pd.read_csv(r'MATCHdateWITHname.csv')

labels = df.LABEL

la = []
for i in range(88):
	la.append(labels[i][26:34])
	 
	# print(labels[i[:8])
li = []

for i in range(len(df.dates)):
	# print(i)
	if df.names[i][:8] in la:
		li.append(df.dates[i])

df2 = pd.DataFrame(li) 
df2.to_csv(r'MATCHEDdateWITHname.csv')
# pd.to_csv(r'MATCHEDdateWITHname.csv', df2))