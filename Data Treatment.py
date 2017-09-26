import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Step 0: Accessing the training and testing sets 

train = pd.read_csv('...train.csv')
test = pd.read_csv('...test.csv')

# Combining training and testing set to obtain an overall look at the data set 

result = [train, test] 
together = pd.concat(result, join_axes=None, ignore_index=True,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True)

# Step 1: Party Size Treatment 

# Note 1: the variable SibSp represents the total number of siblings + spouse they are travelling with 
# Note 2: the variable Parch represents the total number of parents + children they are travelling with 

##### OBJECTIVE
# Find the number of people they are travelling with. This includes non sibling/spouse companions. 
# Find the maximum number between (parch + sibsp + 1) and the number of times their ticket number appears in the concatenated list between training and testing set. 
# (Everyone in the same travelling party has the same ticket number) 

for i, row in together.iterrows():
    parch = together.ix[i, 'Parch']
    sibsp = together.ix[i, 'SibSp']
    ticket = together.ix[i, 'Ticket']
    temp = together['Ticket'].value_counts()
    together.ix[i, 'PartySize'] = max(temp[ticket], parch + sibsp + 1)

# Bucket party size into 0 companions, 2-4 companions, 5+ companions 

for i, row in together.iterrows():
    if together.ix[i, 'PartySize'] == 0: 
        together.ix[i, 'PartySize1'] = '0'
    elif together.ix[i, 'PartySize'] == 1: 
        together.ix[i, 'PartySize1'] = '1'
    elif together.ix[i, 'PartySize'] in (2,3,4): 
        together.ix[i, 'PartySize1'] = '2 to 4'
    else: 
        together.ix[i, 'PartySize1'] = '5+'


# Step 2: Seperating Sibling and Spouse 

#### OBJECTIVE: create two new binary variables wwhich indicates if the passenger has a spouse or has at least one sibling 
# Find the different types of name titles. Use titles and SibSp variable to determine if they have a sibling/spouse. 
# SibSp is the sum of sibling(s) and spouse they are travelling with. 

titles = [] 

for i, row in together.iterrows(): 
    if re.findall(', [a-zA-Z]+.', together.ix[i, 'Name']) not in titles: 
        titles.append(re.findall(', [a-zA-Z]+.', together.ix[i, 'Name']))

# These are the titles: Mr, Mrs, Miss, Master, Don, Rev, Dr, Mme, Ms, Major, Lady, Sir, Mlle, Col, Capt, the, Jonkheer, Dona 
# Has Spouse determination (female): Has title "mrs/mme/lady" and SibSp > 0 -> has spouse; otherwise -> unmarried 
# Has Spouse determination (male): if travelling with a married female companion (validated through ticket number) who share their last name and SibSp > 0 -> married; otherwise -> unmarried 
# Has Sib determination: subtract one from SibSp if they are married. If remaining SibSp count > 0, then they have a sibling. 

# Note: as HasSib and HasSpouse (HasSp) tells a similar story as SibSp and Parch, I chose to exclude SibSp and Parch from the final model in favor of HasSib and HasSpouse (HasSp)

# Female

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

# Male

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

# Step 3: Creating Age Buckets 

# Visualize changes in survival rate against age 

age = train['Age'].drop_duplicates()
survival_by_age = {}

for i in age: 
    survival_rates = train.ix[train['Age'] == i]
    survival_by_age[i] = float(sum(survival_rates['Survived']))/float((max(len(survival_rates), 1)))

lists = sorted(survival_by_age.items())

x,y = zip(*lists)

plt.scatter(x,y)
plt.show()

# Based on graph, split ages accordingly: 15 or younger, 15 - 60 and over 60
# Unknown ages are grouped in 15-60 if married; 15 and under otherwise

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
        elif together.ix[i, 'HasSib'] == 1: 
            together.ix[i, 'Age1'] = '15 and under'
        else: 
            together.ix[i, 'Age1'] = 'over 60'
 
 # Step 4: Determine if they are a crew member   
          
# Crew members pay no fare 
# Death rates of crew member significantly higher than regular passengers

for i, row in together.iterrows():
    if together.ix[i, 'Fare'] == 0: 
        together.ix[i, 'IsCrew']= 1
    else: 
        together.ix[i, 'IsCrew'] = 0 

# Step 5: Check for feature correlations 

# Possible interaction effects to be considered in model selection: 
# PartySize x HasSib, PClass x Fare

corr = together.corr() 
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# Split treated data back into train and test for model training and testing  

temp = pd.concat([train.set_index('PassengerId'), together.set_index('PassengerId')], axis = 1, join = 'inner').reset_index()
temp2 = temp[['Survived', 'PartySize1', 'IsCrew', 'Sex', 'Age1', 'HasSp', 'HasSib', 'Pclass', 'Embarked']]
treated_train = temp2.loc[:, ~temp2.columns.duplicated()]

temp_test = pd.concat([test.set_index('PassengerId'), together.set_index('PassengerId')], axis = 1, join = 'inner').reset_index()
temp_test2 = temp_test[['PassengerId', 'PartySize1', 'IsCrew', 'Sex', 'Age1', 'HasSp', 'HasSib', 'Pclass', 'Embarked']]
treated_test = temp_test2.loc[:, ~temp_test2.columns.duplicated()]

# Fitting the model 

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from patsy import dmatrices, dmatrix
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier 

# Step 6: creating dmatrices

# Elected not to include IsCrew due to the noise it creates (not enough crew members to justify the noise) 
# Elected not to include any interaction effects due to noise 

y, X = dmatrices('Survived ~ C(HasSp) + C(HasSib) + C(Embarked) + C(Age1) + C(Pclass) + C(Sex) + C(PartySize1)', treated_train, return_type = 'dataframe')

# Step 7: 10 fold cross validation 
# GBM performed the best at about an average of ~83%

n = len(treated_train) 
kf = KFold(n_splits = 10) 
kf.get_n_splits(X)  

logreg_performance = []
rf_performance = [] 
svc_performance = []
gbm_performance = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.ix[train_index], X.ix[test_index]
    y_train, y_test = y.ix[train_index], y.ix[test_index]
    
    # Logistic Regression 
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    log_pred = logreg.predict(X_test)   
    logreg_score = accuracy_score(y_test, log_pred) 
    logreg_performance.append(logreg_score) 
    
    # Random Forest 
    rf = RandomForestClassifier(n_estimators = 100) 
    rf.fit(X_train, y_train) 
    rf_pred = rf.predict(X_test) 
    rf_score = accuracy_score(y_test, rf_pred) 
    rf_performance.append(rf_score) 
    
    # Support Vector Machine 
    svc = SVC() 
    svc.fit(X_train, y_train) 
    svc_pred = svc.predict(X_test) 
    svc_score = accuracy_score(y_test, svc_pred) 
    svc_performance.append(svc_score)
    
    # GBM 
    gbm = GradientBoostingClassifier()
    gbm.fit(X_train, y_train) 
    gbm_pred = gbm.predict(X_test)
    gbm_score = accuracy_score(y_test, gbm_pred)
    gbm_performance.append(gbm_score)

# Step 8: Fitting the entire model 

gbm.fit(X, y)
final_test = dmatrix('~ C(HasSp) + C(HasSib) + C(Embarked) + C(Age1) + C(Pclass) + C(Sex) + C(PartySize1)', treated_test, return_type = 'dataframe')
final_pred = logreg.predict(final_test) 
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": final_pred})
submission.to_csv('...Python_submission.csv', index=False)