
import pandas as pd 
Dates = pd.read_csv(r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment26_0612020\liststd1in1.csv')
Length = pd.read_csv(r'F:\Experiments\Crystal Pendant\TestExp26_associateTimeAndFrame\CrystalL.csv')

number = pd.DataFrame()
# print(Dates.iloc[:,])
# number = pd.to_numeric(Length['Label'].str.slice(start=25))
number['Label'] = Length['Label'].str.slice(start=25)

# print (number['Label'], Dates.iloc[:,0])
# print('dates=', Dates.iloc[:,0])
slice = []
time = []
pict = []

# dft = number[:,0] == Dates.iloc[:,0]
# for i in range(0,700,1):
# 	if str(i) in number['Label']:
# 		print ('i=',i)
# 		if i in Dates.iloc[:,0]:
# 			slice.append(i)
# 			time.append(Dates.iloc[i,1])
# 			pict.append(Dates.iloc[i,2])

for i in number['Label']:
	i = int(i)//2
	print(i)
	print(Dates.iloc[i,0])

	slice.append(2*i)
	time.append(Dates.iloc[i,1])
	pict.append(Dates.iloc[i,2])
	# print ('i2=',i)		
df = pd.DataFrame(list(zip(slice, time,pict)), 
               columns =['slice', 'time','pict']) 

# print(df['time'],df['pict'])
print(df)			
# 			
df.to_csv(r'F:\Experiments\Crystal Pendant\TestExp26_associateTimeAndFrame\zip.csv')
