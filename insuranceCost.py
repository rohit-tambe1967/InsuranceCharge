# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:30:43 2019

@author: COMPAQ
"""

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr

from sklearn import linear_model

import statsmodels.api as sm
from statsmodels.formula.api import ols

"""
# Plotly Packages
from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
"""

path = 'D:\\DataScience\\InsuranceCost\\'

datafile = path + 'insurance.csv'

dfData = pd.read_csv(datafile)

#- Make a backup copy of the data
dfOrignal = dfData.copy()
dfOrignal.head()


#-- understand the data
dfData.info
dfData.columns   #- ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'] 

dfData.count(axis=0) 
"""
age         1338
sex         1338
bmi         1338
children    1338
smoker      1338
region      1338
charges     1338
"""

dfData.isnull().sum(axis=0)
"""
age         0
sex         0
bmi         0
children    0
smoker      0
region      0
charges     0
"""

dfData.describe(include = 'all')
dfData.describe(include = 'all').age

#---

##- Check for blank values in each column

#--- Check for Age
dfData[np.isnan(dfData.age)]
dfData[np.isnan(dfData.age)].age.count()  #- Shows 0 , means no blank in Age col

#--- Check for Sex
dfData.isnull()
dfData[dfData.sex.isnull()].sex.count()  #- Shows 0, means no blank sex in any row

dfData.sex.unique()  # - Shows Unique values in Sex col. -> ['female', 'male']

#-- Check for BMI 
dfData[dfData.bmi.isnull()].bmi.count()  #- 0 rows have null BMI 

#-- Check for Children
dfData.children.unique()    #- [0, 1, 3, 2, 5, 4]  , means there are no nulls

#-- check for smoker
dfData.smoker.unique()      #- ['yes', 'no'] 
#--- Make smoker flag a Boolean col to use in reg analysis
dfData['nSmoker'] = dfData['smoker'].apply( lambda x: 1 if x== 'yes' else 0 )
dfData.head(20)
#--- 
#- check for blank region
dfData.region.unique()  #- ['southwest', 'southeast', 'northwest', 'northeast']



#-- Plot some charts to view the data distribution

#- Age distribution
"""
dfData['age'].values
Out[12]: array([19, 18, 28, ..., 18, 21, 61], dtype=int64)

[dfData['age'].values]
Out[13]: [array([19, 18, 28, ..., 18, 21, 61], dtype=int64)]

type([dfData['age'].values])
Out[14]: list

type(dfData['age'].values)
Out[15]: numpy.ndarray
"""

sns.boxplot(dfData['age'])
sns.distplot(dfData['age'].values, kde=True, rug=True, bins=4 )

# lstAge = [dfData['age'].values.tolist()]
dfData.age.describe()
"""
count    1338.000000
mean       39.207025
std        14.049960
min        18.000000
25%        27.000000
50%        39.000000
75%        51.000000
max        64.000000
"""

""" 
Make 4 age groups  as - 
Young  - Age 18 - 32 yrs
Middle - Age 33 - 44 yrs
Senior - Age 45 - 56 yrs
Elder -  Age > 56 yrs

"""
dfData['ageGrp'] = np.nan   #- add a new  category column for age group
lstAge = [dfData]   # copy the data to list 
for x in lstAge:
    x.loc[(x['age'] <= 32), 'ageGrp'] = 'Young'
    x.loc[(x['age'] >= 33) & (x['age'] <= 44), 'ageGrp'] = 'Middle'
    x.loc[(x['age'] >= 45) & (x['age'] <= 56), 'ageGrp'] = 'Senior'
    x.loc[(x['age'] > 56), 'ageGrp'] = 'Elder'

sns.countplot(x=dfData.region, hue=dfData.ageGrp )
#- The regionwise age distribution is almost equal across all regions

##--

##-- Analyse the 'charges'
sns.boxplot(dfData['charges'])
sns.boxplot(np.log(dfData['charges']))

dfData.charges.describe()
sns.distplot(dfData['charges'].values, kde=True, rug=True, bins=4 )
sns.distplot(np.log(dfData['charges'].values), kde=True, rug=True, bins=4 )

##---

#-- Analyse BMI
dfData.bmi.describe()
sns.boxplot(dfData['bmi'])

sns.distplot(dfData['bmi'].values, kde=True, rug=True, bins=4 )

#- Age group -wise BMI
plt.figure(figsize =(16,4))
plt.subplot(2,2,1)
sns.boxplot(dfData["bmi"].loc[dfData["ageGrp"] == "Young"].values)
plt.subplot(2,2,2)
sns.boxplot(dfData["bmi"].loc[dfData["ageGrp"] == "Middle"].values)
plt.subplot(2,2,3)
sns.boxplot(dfData["bmi"].loc[dfData["ageGrp"] == "Senior"].values)
plt.subplot(2,2,4)
sns.boxplot(dfData["bmi"].loc[dfData["ageGrp"] == "Elder"].values,  )
plt.show()

sns.set(style="ticks", palette="pastel")
g = sns.boxplot(x="ageGrp", y="bmi", hue="region", data=dfData, width=.9)
g.legend(bbox_to_anchor=(1,1))
plt.show()

#- Age groupwise BMI for smoker and non smoker
#sns.set(style="ticks", palette="pastel")
g = sns.lmplot(x="age", y="bmi", hue="nSmoker", data=dfData, 
               fit_reg=True, palette='Set1')
#g.legend(bbox_to_anchor=(1,1))
plt.show()

""" Make 4 weight categories based on BMI
Under Weight (UndrW): (BMI)  <  18.5
Normal Weight (NormW):(BMI)  ≥  18.5 and (BMI)  <  24.9
Overweight (OvrW): (BMI)  ≥  25 and (BMI)  <  29.9
Obese : (BMI)  >  30
"""
dfData['wtype'] = np.nan   #- add a new body type column for bmi group
lstBMI = [dfData]   # copy the data to list 
for x in lstBMI:
    x.loc[(x['bmi'] < 18.5), 'wtype'] = 'UWt'
    x.loc[(x['bmi'] >= 18.5) & (x['bmi'] < 24.9), 'wtype'] = 'NWt'
    x.loc[(x['bmi'] >= 25) & (x['bmi'] < 29.9), 'wtype'] = 'OWt'
    x.loc[(x['bmi'] > 29.9), 'wtype'] = 'Obs'

dfData.head(20)
sns.countplot(x=dfData.region, hue=dfData.wtype, palette="Set1" )

#- Analyse Region assign numbers to regions
dfData.region.unique()  #- ['southwest', 'southeast', 'northwest', 'northeast']

dfData['nReg'] = np.nan   #- add a new  category column for age group
lstReg = [dfData]   # copy the data to list 
for x in lstReg:
    x.loc[(x['region'] == 'northeast'), 'nReg'] = 1
    x.loc[(x['region'] == 'southeast'), 'nReg'] = 2
    x.loc[(x['region'] == 'southwest'), 'nReg'] = 3
    x.loc[(x['region'] == 'northwest'), 'nReg'] = 4

dfData.head(20)
dfData.loc[:,['region','nReg']].head(20)    #- Show only named columns

#- Regionwise smoker
sns.countplot(x=dfData.region, hue=dfData.smoker, palette="Set1" )


#--- 



#-- Find correlation 

#-- Age & BMI
lmAgeBMI = ols('bmi ~ ageGrp', data=dfData).fit()
lmAgeBMI.summary()

np.corrcoef(dfData.bmi, dfData.age) #- [0.10927188]

#- Age & Charges
np.corrcoef(dfData.age, dfData.charges) #- 0.29900819

#- BMI & Charges
np.corrcoef(dfData.bmi, dfData.charges) #- .19834097

#- smoker & BMI
np.corrcoef(dfData.nSmoker, dfData.bmi)  #- 0.00375043 - Almost no correlation

#- smoker & charges
np.corrcoef(dfData.nSmoker, dfData.charges)  #- 0.78725143 - High correlation

#- children and charges
np.corrcoef(dfData.children, dfData.charges) #-  0.06799823 - Very Low

#- Region and Charges
np.corrcoef(dfData.nReg, dfData.charges) #- [-0.0502262]  -ve correl

sns.lmplot(x='bmi', y='charges',  data=dfData, hue='ageGrp', fit_reg=False,  palette="Set1", legend=True)
#plt.legend(loc='lower right')
plt.show()

f, (ax1, ax2, ax3,ax4,ax5) = plt.subplots(ncols=5, figsize=(30,8))
sns.stripplot(x='ageGrp', y='charges',  hue='smoker', data=dfData, ax=ax1, linewidth=1, palette="Set2",
              size=5, jitter=.25, dodge=True)
ax1.set_title('Age vs. Charges')
sns.stripplot(x='ageGrp', y='charges',  hue='wtype', data=dfData, ax=ax2, linewidth=1, palette="Set1", 
              size=5, jitter=.25, dodge=True)
ax2.set_title('Wt., Age vs. Charges')
sns.stripplot(x='smoker', y='charges',  hue='wtype', data=dfData, ax=ax3, linewidth=1.2, palette="Set1",
              size=5, jitter=.25, dodge=True)
ax3.set_title('Wt., smoker vs. Charges')
sns.stripplot(x='region', y='charges', hue='smoker', data=dfData, ax=ax4, linewidth=1.2, palette="Set1",
              size=5, jitter=.25, dodge=True)
ax4.set_title('Region, smoker vs. Charges')
sns.stripplot(x='ageGrp', y='bmi',  hue='smoker', data=dfData, ax=ax5, linewidth=.8, palette="Set2",
              size=5, jitter=.25, dodge=True)
ax5.set_title('Age vs. BMI Vs Smoker')
plt.show()

f,(ax6,ax7) = plt.subplots(ncols=2, figsize=(16,6))
sns.stripplot(x='smoker', y='age',  hue='wtype', data=dfData, ax=ax6, linewidth=.8, 
              palette="Set2", size=5, jitter=.25, dodge=True)
ax6.set_title('Smoker vs. Age Vs Wtype')
sns.stripplot(x='wtype', y='age',  hue='smoker', data=dfData, ax=ax7,
              linewidth=.8, palette="Set2", size=5, jitter=.25, dodge=True)
ax6.set_title('Wtype vs. Age Vs Smoker')
plt.show()

#-- Plot a 3D scatterplot along age, wt, smoker
scat3d = plt.figure()
ax = scat3d.add_subplot(111, projection='3d')

xAxis = [dfData.age.values]
zAxis = [dfData.bmi.values]
yAxis = [dfData.nSmoker.values]

ax.scatter(xAxis,yAxis,zAxis,  marker='x')
ax.set_xlabel('Age')
ax.set_ylabel('Smoker')
ax.set_zlabel('BMI')
plt.show()

#---

#- Regression 1
X1 = dfData[['age','bmi','nSmoker','children']]
X1.head()
y1 = dfData[['charges']]
y1.head()

regr1 =  linear_model.LinearRegression()
regr1.fit(X1, y1)

print("Intercept: " , regr1.intercept_) #- [-12102.76936273]
print("Coeff.: " , regr1.coef_) #- [  257.84950728   321.85140247 23811.3998446    473.50231561]

regr1.predict([[35,25,1,1]])
#Out[66]: array([[29253.1506138]])

regr1.predict([[37,34.2,1,1]])
#Out[67]: array([[32729.88253104]])

regr1.predict([[37,34.2,0,1]])
#Out[68]: array([[8918.48268643]])

regr1.predict([[37,25,0,1]])
#Out[69]: array([[5957.44978375]])

regr1.predict([[37,25,1,1]])
#Out[70]: array([[29768.84962835]])

#- Regression 2

X2 = dfData[['age','bmi','nSmoker']]
X2.head()
y2 = dfData[['charges']]
y2.head()

regr2 =  linear_model.LinearRegression()
regr2.fit(X2, y2)

print("Intercept: " , regr2.intercept_) #- [-11676.83042519]
print("Coeff.: " , regr2.coef_) #- [  259.54749155   322.61513282 23823.68449531]

regr2.predict([[37,34.2,1]])
#Out[12]: array([[32783.54879995]])

regr2.predict([[37,34.2,0]])
#Out[13]: array([[8959.86430464]])


