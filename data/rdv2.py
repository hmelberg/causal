# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:23:30 2015

@author: hmelberg_user
"""


# The question: Does class size affect student achievement?
# Cannot simply compare average outcomes in small and large classes 
# For instance, rich schools may have small classes and do well for reason other than class size
# One solution: Use a method which compares outcomes in classes of different size when this difference is caused by external factors unrelated to outcome
# Example: Use the rule that a school cannot have more than 32 students in a class 
# This is a discrete cutoff. We can compare outcomes in classes

#%% imports

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


#%% read data
#df = pd.read_stata(r"""C:\Users\hmelberg_user\Google Drive\data\div/angrist.dta""")

df = pd.read_stata(r"""http://www.ats.ucla.edu/stat/stata/examples/methods_matter/chapter9/angrist.dta""")



# The Angrist & Lavy Dataset includes school level data for:
# size: fifth-grade cohort size
# intended_classsize: average intended class size for each school
# observed_classize: observed average class size for each school
# read: average reading achievement in cohort

#%% basic descriptive
df = df.rename(columns = {'size': 'cohort_size'})





df = df.rename(columns = {'size': 'cohort_size'})

df.head()
df.columns
df.dtypes
df.describe()

#%% more descriptives (note df.size.plot does not work since size is a reserved pandas method, so not a bug but still)
df.read.plot(kind = "hist")
df.cohort_size.plot(kind = "hist") 

sns.corrplot(df)

plt.scatter(df['cohort_size'], df['observed_classize'])
plt.scatter(df['cohort_size'], df['intended_classize'], c ="r")




df.groupby('cohort_size')['observed_classize', 'intended_classize', 'read'].mean().plot(xlim=(35,45), figsize = (10,8))












df.groupby('cohort_size')['observed_classize'].mean().plot()

df.groupby('cohort_size')[['observed_classize', 'intended_classize']].mean().plot()

df.groupby('cohort_size')[['intended_classize', 'read']].mean().plot()


df.groupby('cohort_size')['read'].mean().plot(xlim = (30,50))
df.groupby('cohort_size')['read'].mean().plot(xlim = (70,90))



#%% Naive analysis (first)

ols_model = "read ~ observed_classize"

ols_analysis = sm.formula.ols(ols_model, data = df).fit()
ols_analysis.summary()

#%% Regression Discontinuity

#%% Sharp, window approach

df2 = df[(df.observed_classize > 35) & (df.observed_classize < 45 )]


df2.describe()

df2['above40'] = 0

df2['above40'][df2.cohort_size > 39] = 1

ols_model = "read ~ observed_classize + above40"

ols_analysis = sm.formula.ols(ols_model, data = df2).fit()
ols_analysis.summary()






df['class_size_from_threshold'] = df.cohort_size - 41

df['above40'] = 0
df['above40'][df.class_size_from_threshold > 0] = 1


step1_regression = sm.formula.ols(step1_model, data = df).fit()
step1_regression. summary()


# limit analysis to 29 to 53

df2 = df[(df.cohort_size) > 28 & (df.cohort_size > 54)]

model = "read ~ class_size_from_threshold + above40"
rd_sharp_window = sm.formula.ols(model, data = df2).fit()
rd_sharp_window.summary()




# Step 1: Class size as a function of cohort size
step1_model = "observed_classize ~ cohort_size"
step1_regression = sm.formula.ols(step1_model, data = df).fit()
step1_regression. summary()
df['predicted'] = step1_regression.predict()

# Step 2: Reading score as a function of predicted class size
step2_model = "read ~ predicted"
step2_regression = sm.formula.ols(step2_model, data = df).fit()
step2_regression. summary()



df['predicted'] = step2_regression.predict()




# Create variables

# First: Center variable around cutoff (41) i.e. make 41 the zero point










plt.scatter(df.cohort_size, df.read)
plt.xlim(30,50)
df.groupby('observed_classize')['read'].mean().plot()

avg = df.groupby('cohort_size')[['observed_classize', 'intended_classize', 'read']].mean()
plt.scatter( [avg.index, avg.index, avg.index], [avg.observed_classize, avg.intended_classize, avg.read])
plt.xlim(20,50)

df['cohort_size_centered'] = df.size - 41





def small(size):
    if(size>=41):
        return 1
    return 0
    
# "first" distinguishes the groups that participate in the first diff.
def first(group):
    groups = {1: 0, 2:0,
              3: 1, 4:1}
    return groups[group]

# SET UP Forcing Variable and Cutoff Predictor
class_df['small'] = class_df['size'].map(small)
class_df['csize'] = class_df['size'].map(lambda x: x-41)

# summarize the read variable by each class size group
class_df[(class_df['size']>=36) & (class_df['size']<=46)].boxplot("read", "csize")
plt.show()

