# take-home challenge tips:
# data&inspect, data preprocessing, model&explaination

import cPickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix

# data&inspect
data=pd.read_csv('data.csv')
data.describe() # only for numeric variables
data.dtypes # check data types
data.head(5)

data.loc[data.col1 = ..., :] # categorical var - check outliers
data.col1[data.col1 != ...,:].means() # check if remove outliers
data.col1[data.col1 != ...,:].describe() # check if remove outliers
data_1=data.loc[data.col1 != ...;:] # remove outliers if that makes sense

col2_dist = data_1.col2.value_counts() # print the dist for col2
col2_dist.plot(kind='bar') # or kind = 'hbar'
col3_dist = data_1.group_by('col1').col3.mean() # print mean col3 grouped by col1
col3_dist.plot(kind='bar')

plt.scatter(data_1.col4, data_1.target) # scatter plot(numeric var) - checking outliers

# insights or problems can be found from above
# what should we do next??

# data preprocessing
data.time1 = pd.to_datetime(data.time1) # deal with date variable
data.time_diff = (data.time1-data.time2).dt.total_seconds() # dt.weekday_name to get weekday names

data.drop(['var1','var2'], axis=1, inplace=True) # drop variables
del data.var1

var_dist = data.var1.value_counts()
data['var1_dist'] = data.var1.map(var_dist) # map the dist var to every var1

data['is_male'] = (data.sex=="M").astype(int) # categorical var to binary var

data.test_var.replace(' ', '_', inplace=True, regex=True) # replace space by _

data.columns=data.columns.str.replace(' ','_') # replace space by _ for all vars

data.rename(columns={'var':'new_var_name'}, inplace=True) # rename vars

pages = ["home_page","search_page","payment_page","payment_confirmation_page"]
data["final_page"] = data.final_page.astype("category",categories = pages,ordered=True) # categorical var w/ order
data["final_page"] = data.final_page.astype("category",categories = pages,ordered=False) # categorical var w/o order

data.loc[(data.var1==' '), 'var1']=0 # missing to 0

data[var1]=pd.to_numeric(data.var1) # object to numeric var

from sklearn.preprocessing import LabelEncoder
data['var1_encoded']=LabelEncoder().fir_transform(data.var1) # encode vars

data.var1=np.where(data.vars=="M", 'Y','N') # if statement..

weekday2index = {"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}
data["weekday"] = data.weekday.map(weekday2index) # map function!!

data.to_csv('dataname.csv', index_label=var1) # save to csv

data_X = pd.get_dummies(data.loc[:, ('var1','var2','var3',...)])
## or
## data_X = pd.get_dummies(data.loc[:, data.columns!='target'])
Y = data_1.target
Xtrain,Xtest,ytrain,ytest = train_test_split(data_X,Y,test_size=0.333, random_state=seed_id) # stratify=y to randomized stratify on y
ytrain.mean(),ytest.mean() # check the imbalance between train&test

# model&explaination - Logistic Regression
lrcv = LogisticRegressionCV(Cs = np.logspace(-3,3,7),
                            dual=False,
                            scoring='roc_auc',
                            max_iter=1000,
                            n_jobs=-1,
                            verbose=1)
lrcv.fit(Xtrain,train) # fit model and print score
lrcv.score(Xtrain,ytrain)
lrcv.score(Xtest,ytest)


ytest_predict = lrcv.predict(Xtest) # print score on the prediction dataset
print classification_report(y_true=ytest,y_pred=ytest_predict)

feat_importances = pd.DataFrame({"name":Xtrain.columns,"coef":lrcv.coef_[0]})
feat_importances = feat_importances[['name','coef']]# reorder the columns
feat_importances['importances'] = np.abs( feat_importances['coef'] )
feat_importances.sort_values(by="importances",inplace=True,ascending=False)
feat_importances # check var importance

# conclusions and next-steps (how to improve, what kind of analysis is needed)


# model - xgboost
import xgboost as xgb
train_matrix = xgb.DMatrix(xtrain, train)
test_matrix = xgb.DMatrix(xtest)

params = {} # use cross validation to fit model
params['silent'] = 1
params['objective'] = 'binary:logistic'  # output probabilities
params['eval_metric'] = 'auc'
params["num_rounds"] = 300
params["early_stopping_rounds"] = 30
# params['min_child_weight'] = 2
params['max_depth'] = 6
params['eta'] = 0.1
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8

cv_results = xgb.cv(params,train_matrix,
                    num_boost_round = params["num_rounds"],
                    nfold = params.get('nfold',5),
                    metrics = params['eval_metric'],
                    early_stopping_rounds = params["early_stopping_rounds"],
                    verbose_eval = True,
                    seed = seed)


from sklearn.metrics import accuracy_score,classification_report,roc_curve
def plot_validation_roc(): # roc definition
    """
    we cannot plot ROC on either training set or test set, since both are biased
    so I split the training dataset again into the training set and validation set
    retrain on the training set and plot ROC on the validation set and choose a proper cutoff value
    
    define a class to limit the naming group, avoid polluting the global naming space
    """
    Xtrain_only,Xvalid,ytrain_only,yvalid = train_test_split(Xtrain,ytrain,test_size=0.3,random_state=seed)
    onlytrain_matrix = xgb.DMatrix(Xtrain_only,ytrain_only)
    valid_matrix = xgb.DMatrix(Xvalid,yvalid)

    temp_gbt = xgb.train(params, onlytrain_matrix, n_best_trees,[(onlytrain_matrix,'train_only'),(valid_matrix,'validate')])
    yvalid_proba_pred = temp_gbt.predict(valid_matrix,ntree_limit=n_best_trees)

    fpr,tpr,thresholds = roc_curve(yvalid,yvalid_proba_pred)
    return pd.DataFrame({'FPR':fpr,'TPR':tpr,'Threshold':thresholds})

roc = plot_validation_roc() # plot roc
plt.figure(figsize=(10,5))
plt.plot(roc.FPR,roc.TPR,marker='h')
plt.xlabel("FPR")
plt.ylabel("TPR")

xgb.plot_importance(gbt) # check variable importance

# Let's say you now have this model which can be used live to predict in real-time 
## if an activity is fraudulent or not. From a product perspective, how would you 
## use it? That is, what kind of diﬀerent user experiences would you build based on
## the model output?
# since my model can predict the probability a purchase is a fraud, so I need to set
## two probability cutoffs as 'alert value', alert1 and alert2, and alert1 < 
## alert2.
# for an incoming purchase, my model will return the probability 'p' that the 
## purchase is a fraud,
## if p < alert1, then I assume the purchase is normal, proceed without any 
## problem
## if alert1 <= p < alert2, then I assume the purchase is suspicious, I will ask 
## the customer for additional authorization. for example, send an email or SMS to
## the customer, let him/her authorize the purchase.
## if p>= alert2, then the purchase is highly suspicious, I not only ask the 
## customer for additional authorization via email or SMS, but also put the 
## purchase on hold and send the purchase information to some human expert for 
## further investigation.


# model - tree
from sklearn.tree import DecisionTreeClassifier,export_graphviz
dt = DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,min_samples_split=20)
dt.fit(X,y)
export_graphviz(dt,feature_names=X.columns,class_names=['NotFraud','Fraud'],
                proportion=True,leaves_parallel=True,filled=True)
pd.Series(dt.feature_importances_,index = Xtrain.columns).sort_values(ascending=False)


# check importance - chi2
from sklearn.feature_selection import chi2,f_classif
scores, pvalues = chi2(X,y)
df.DataFrame({'scores':scores, "pvalues":pvalues}, index=X.columns).sort_value(by='scores', ascending=False)


# f scores
from sklearn.feature_selection import chi2,f_classif
fscores,_ = f_classif(X,y)


# Other functions:
pd.concat([df1, df2], keys=['df1','df2'], axis=1) # concat 2 dfs
df.var1.astype(np.float) # convert to float
df.groupby('var1').var2.apply(lambda x: s.value_counts(normalize=True)).unstack()
df.var1.agg(['count','mean']).sort_values(by='var2', ascending=True)

df.plot()
df.transpose().plot()

print "{:.2f}% of users opened the email".format(100)

df.fillna(value=0, inplace=True)
