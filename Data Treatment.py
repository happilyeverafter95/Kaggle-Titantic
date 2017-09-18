import pandas as pd 
import matplotlib.pyplot as plt
import re

# Step 0 part 1: Accessing the training and testing sets 

train = pd.read_csv('C:\\Users\\mandy\\Desktop\\Kaggle competitions\\Titanic\\train.csv')
test = pd.read_csv('C:\\Users\\mandy\\Desktop\\Kaggle competitions\\Titanic\\test.csv')

#Combining training and testing set to obtain an overall look at the data set 

result = [train, test] 
together = pd.concat(result, join_axes=None, ignore_index=True,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True)

# Step 1: treatments and transformations. 
# Find the number of people they are travelling with. This includes non sibling/spouse copmanions. 
# Find the maximum number between parch + sibsp + 1 and the number of occurrences of the ticket number. 

for i, row in together.iterrows():
    parch = together.ix[i, 'Parch']
    sibsp = together.ix[i, 'SibSp']
    ticket = together.ix[i, 'Ticket']
    temp = together['Ticket'].value_counts()
    together.ix[i, 'PartySize'] = max(temp[ticket], parch + sibsp + 1)

# Find the different types of titles. Using titles to seperate spouse and sibling in SibSp. 

titles = [] 

for i, row in together.iterrows(): 
    if re.findall(', [a-zA-Z]+.', together.ix[i, 'Name']) not in titles: 
        titles.append(re.findall(', [a-zA-Z]+.', together.ix[i, 'Name']))

# These were the titles: Mr, Mrs, Miss, Master, Don, Rev, Dr, Mme, Ms, Major, Lady, Sir, Mlle, Col, Capt, the, Jonkheer, Dona 
# Has Spouse determination (male): if travelling with a female companion (validated through ticket number) who is married with their last name -> married; otherwise -> unmarried 
# Has Spouse determination (female): Has title "mrs/mme/lady" and SibSp > 0 -> has spouse
# Has Sib determination: what is left from SibSp after doing spouse reduction 

for i, row in together.iterrows(): 
    if ('Mrs.' in together.ix[i, 'Name'].split() or 'Mme' in together.ix[i, 'Name'].split() or 'Lady' in together.ix[i, 'Name'].split()) and together.ix[i, 'SibSp'] > 0 : 
        together.ix[i, 'HasSp'] = 1
        if together.ix[i, 'SibSp'] > 1: 
            together.ix[i, 'HasSib'] = 1
        else: 
            together.ix[i, 'HasSib'] = 0 
    elif together.ix[i, 'Sex'] == 'female': 
        together.ix[i, 'HasSp'] = 0
        if together.ix[i, 'SibSp'] > 0: 
            together.ix[i, 'HasSib'] = 1
        else: 
            together.ix[i, 'HasSib'] = 0

for i, row in together.iterrows():
    if together.ix[i, 'Sex'] == 'male' and together.ix[i, 'SibSp'] >0: 
        ticket = together.ix[i, 'Ticket'] 
        companions = together.ix[(together['Ticket'] == ticket) & (together['Sex'] == 'female') & (together['HasSp'] == 1)]
        name = together.ix[i, 'Name'].split()
        HasSp = False
        for j, row in companions.iterrows(): 
           if name[0] == companions.ix[j, 'Name'].split()[0]:
               together.ix[i, 'HasSp'] = 1
               HasSp = True 
        if HasSp == False:
            together.ix[i, 'HasSp'] = 0
            together.ix[i, 'HasSib'] = 1
        else:
            together.ix[i, 'HasSib'] = 1 
    elif together.ix[i, 'Sex'] == 'male' and together.ix[i, 'SibSp'] == 0: 
        together.ix[i, 'HasSp'] = 0
        together.ix[i, 'HasSib'] = 0 

# Creating Age Buckets. Visualize changes in survival_rates through age. 

age = train['Age'].drop_duplicates()
survival_by_age = {}

for i in age: 
    survival_rates = train.ix[train['Age'] == i]
    survival_by_age[i] = float(sum(survival_rates['Survived']))/float((max(len(survival_rates), 1)))

import matplotlib.pylab as plt

lists = sorted(survival_by_age.items())

x,y = zip(*lists)

plt.scatter(x,y)
plt.show()

# Based on graph, split ages based on: 15 or younger, 15 - 60 and over 60  
for i, row in together.iterrows():
    if together.ix[i, 'Age'] > 0 and together.ix[i, 'Age'] <= 15: 
        together.ix[i, 'Age1'] = '15 and under'
    elif together.ix[i, 'Age'] > 15 and together.ix[i, 'Age'] < 60: 
        together.ix[i, 'Age1'] = '15 to 60'
    elif together.ix[i, 'Age'] >= 60: 
        together.ix[i, 'Age1'] = 'over 60'
    else: 
        if together.ix[i, 'HasSp'] == 1: 
            together.ix[i, 'Age1'] = '15 to 60'
        else: 
            together.ix[i, 'Age1'] = '15 and under'
    
# Determine if they are a crew member (crew members pay no fare) 
for i, row in together.iterrows():
    if together.ix[i, 'Fare'] == 0: 
        together.ix[i, 'IsCrew']= 1
    else: 
        together.ix[i, 'IsCrew'] = 0 

# Bucket party size 
for i, row in together.iterrows():
    if together.ix[i, 'PartySize'] == 0: 
        together.ix[i, 'PartySize1'] = '0'
    elif together.ix[i, 'PartySize'] == 1: 
        together.ix[i, 'PartySize1'] = '1'
    elif together.ix[i, 'PartySize'] in (2,3,4): 
        together.ix[i, 'PartySize1'] = 'a2-4'
    else: 
        together.ix[i, 'PartySize1'] = '5+'

# Check for correlations 

plt.matshow(together.corr())

# Split treated data back into train and test for model training and testing  

treated_train = pd.concat([train.set_index('PassengerId'), together.set_index('PassengerId')], axis = 1, join = 'inner').reset_index()
treated_train2 = treated_train[['Survived', 'PartySize1', 'IsCrew', 'Sex', 'Age1', 'HasSp', 'HasSib', 'Pclass', 'Embarked']]

test1 = pd.concat([test.set_index('PassengerId'), together.set_index('PassengerId')], axis = 1, join = 'inner').reset_index()
test2 = test1[['PassengerId', 'PartySize1', 'IsCrew', 'Sex', 'Age1', 'HasSp', 'HasSib', 'Pclass', 'Embarked']]
