import pandas as pd
import numpy as np
# from scipy import integrate
import matplotlib
import os
import seaborn as sns
import matplotlib.pyplot as plt
 

def withoutExtension(string):
    return int(string[:-6])
# Function to calculate Chi-distance
def chi2_distance(A, B):
 
    # compute the chi-squared distance using above formula
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)
                      for (a, b) in zip(A, B)])
 
    return chi
path = r'F:\Experiments\Crystal Pendant\FocusedAnalysis\Profilometer\Profiles crystals at surface\surface crystal profiles - Copy'

x = np.linspace(0,0,500)
y = np.linspace(-2500,0,500)

os.chdir(path) 
names = os.listdir()
names.sort(key=withoutExtension)
print(names)
# main function
# if __name__== "__main__":

fic = [r'F:\Experiments\Crystal Pendant\FocusedAnalysis\Profilometer\Profiles crystals at surface\surface crystal profiles - Copy\9.4.csv',
      r'F:\Experiments\Crystal Pendant\FocusedAnalysis\Profilometer\Profiles crystals at surface\surface crystal profiles - Copy\186.6.csv']
for name in names:

    data = pd.read_csv(name)
    np.array



    a = np.array(data)
    # print(a[1]/)
    b = np.array(pd.read_csv(fic[1], delimiter = ','))

    result = chi2_distance(a, b)
    print("The Chi-square distance for " + name[:-4] + " is :", result)

