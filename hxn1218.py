#	Author: Hasitha Nekkalapu
#	ID: hxn1218
#	DataMining- WineData Classification  

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score

#read from the csv file and return a Pandas DataFrame.
data = pd.read_csv('wine.csv')

#check for duplicates and remove if there are any
data.drop_duplicates(keep = False)
print(data.shape)

#Shuffle the data to ensure randomness
shuffle(data)

#print headers(column names) of the dataset
print("\nThe wine dataset headers are:")
original_headers = list(data.columns.values)
print(original_headers)

# Pandas DataFrame allows you to select columns. 
# We use column selection to split the data into features and class.
# quality is the class attribute we are predicting.  
class_features = data.drop('quality',axis=1)
class_name = data.quality

#Fine tuning the parameters by Recursive Feature Elimination
LRmodel = LogisticRegression()
rfe = RFE(LRmodel, 9)#selecting 9 most important features
fit = rfe.fit(class_features, class_name)
support = rfe.get_support(indices=True)
print(("\nNumber of Features: %d") % fit.n_features_)
print(("Feature Ranking: %s") % fit.ranking_)
print(("Selected Features and their importance: \n %s") % sorted(zip(map(lambda x: round(x, 4), rfe.get_support(indices=True)), original_headers), reverse=True))
selected_features = []
for i in support :
	selected_features.append(original_headers[i])
print("\nSelected features based on their importance are: %s" % selected_features)

#update the features that are important
class_features = data[selected_features]

#preprocessing: making data in range of -1 to 1
class_features = preprocessing.scale(class_features)

#slpitting data into train and test
train_feature, test_feature, train_class, test_class = train_test_split(class_features, class_name,train_size=0.75,test_size=0.25)

#Using RandomForestClassifier to Classify the training data
randomforest = RandomForestClassifier(random_state=0, n_estimators = 50)
#Fitting: Training the ML Algorithm/model
randomforest.fit(train_feature, train_class)

#Obtaining the confidence score for RFC
confidence = randomforest.score(test_feature, test_class)
print("\nTraining set score: {:.3f}".format(randomforest.score(train_feature, train_class)))
print("Test set score: {:.3f}".format(confidence))
print("\nThe confidence score:")
print(confidence)

#predicting the foercasts
prediction = randomforest.predict(test_feature)
print("\nTest set predictions:\n{}".format(prediction))

#Confusion Matrix
print("\nConfusion matrix:\n")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))
print("\n")

#k-fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=10, random_state = None, shuffle = True)
skf.get_n_splits(class_features)

#initializing Average Score
avgScore = []
for train_index, test_index in skf.split(class_features, class_name):
    train_feature, test_feature = class_features[train_index], class_features[test_index]
    train_class, test_class = class_name[train_index], class_name[test_index]

    #Training the RandomForest model
    randomforest.fit(train_feature, train_class)

    #Obtaining the CrossValidation score for RFC
    #append accuracy of each fold to avgScore
    acc = randomforest.score(test_feature, test_class)
    avgScore.append(acc)
    #print("\nTest set predictions:\n{}".format(randomforest.predict(test_feature)))
    print("The accuracy in each of the folds is : {:.2f}".format(acc))

#Calculate the mean CrossValidation Score across k-folds
avgScore = sum(avgScore) / float(len(avgScore))
print("\nAverage accuracy across all folds is: {:.2f}".format(avgScore))
print("improved accuracy is: {:.2f}".format(avgScore - confidence))