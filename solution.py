import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from sklearn.feature_selection import RFE, SelectKBest, f_regression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.metrics import r2_score

SEED = 400


#Recursive Feature Elimination
def RFE_selector(X_train,X_test,y,num_features):
    estimator = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, min_samples_leaf=15, min_samples_split=10,random_state=SEED)
    rfe_selector = RFE(estimator=estimator, n_features_to_select=num_features, step=0.1, verbose=1)
    x_train_rfe = rfe_selector.fit_transform(X_train, y)
    x_test_rfe = rfe_selector.transform(X_test)
    rfe_support = rfe_selector.get_support()
    rfe_feature = x_train.loc[:,rfe_support].columns.tolist()
    return rfe_support, rfe_feature, pd.DataFrame(x_train_rfe),pd.DataFrame(x_test_rfe)


#Filling missing values
def miss_val(x_train, x_test):
    #Fill-in nan values with median of the feature
    x_train = x_train.fillna(x_train.median())
    x_test = x_test.fillna(x_test.median())

    return x_train, x_test

#Remove outliers
def outliers_IF(x_train,y):
    clf = IsolationForest(max_samples=0.99, random_state = SEED, contamination= 0.02)
    outliers = clf.fit_predict(x_train)

    print("Num lines before drop : ", x_train.shape)
    print("Num lines before drop : ", y.shape)
    remove = np.argwhere(outliers == -1)
    for num,line in enumerate(remove):
        if not line == 1:
            x_train = x_train.drop([num])
            y = y.drop([num])

    print("Num lines after drop : ", x_train.shape)
    print("Num lines after drop : ", y.shape)

    return x_train, y

def OutlierDetectionEuclideanMetric(data, ydata):
    neigh = NearestNeighbors(n_neighbors=15)
    neigh.fit(data)

    distancesToNN, indices = neigh.kneighbors(data, return_distance=True)
    meanDistanceOfNN = np.mean(distancesToNN, axis = 1)

    factor = 1.05
    mean_dist_tot = np.mean(meanDistanceOfNN)
    threshold = factor * mean_dist_tot
    index_to_be_removed = meanDistanceOfNN > threshold
    print(index_to_be_removed)
    i=0
    for num in range(data.shape[0]):
        if index_to_be_removed[i] :
            data = data.drop([num])
            ydata = ydata.drop([num])
        i +=1


    print("Num lines after drop : ", data.shape)
    print("Num lines after drop : ", ydata.shape)

    return data, ydata


if __name__ == '__main__':

    x_train_origin =  pd.read_csv("x_train.csv")
    y_train_origin = pd.read_csv("y_train.csv")
    x_test =  pd.read_csv("x_test.csv", delimiter=",", index_col='id')

    #formatting
    train_data = pd.merge(left=x_train_origin, right=y_train_origin, how='inner').drop(columns=['id'])
    x_train = x_train_origin.drop(columns=['id'])
    y = y_train_origin['y']

    #Imputation missing values
    x_train, x_test = miss_val(x_train, x_test)
    #Remove outliers : Isolation Forest
    x_train, y = outliers_IF(x_train, y)
    #x_train, y = OutlierDetectionEuclideanMetric(x_train, y)

    #Feature Selection :
        ####SelectKBest
    #featureSelection = SelectKBest(f_regression, k=100)
    #X_train = featureSelection.fit_transform(x_train, y)
    #scores = featureSelection.scores_
    #print("Shape after feature selection: ", X_train.shape)
        ####RFESelector
    rfe_support,_,x_train,x_test = RFE_selector(x_train,x_test,y,100)
    print ("Train shape after RFESelector: {} ".format(x_train.shape))


    #Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x_train, y, test_size=0.33, random_state=SEED)


    #Regressors :
        ###Gradient Boosting Regressor
    gBoost = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1,
                                   max_depth=4, max_features=0.1 ,
                                   min_samples_leaf=15, min_samples_split=10,
                                   random_state =SEED)

    gBoost.fit(x_train.values, y_train.values)
    y_gBoost_test = gBoost.predict(x_test.values)
    print("Gradient Boosting", r2_score(y_test, y_gBoost_test))

        ###Random Forest
    random_forest = RandomForestRegressor(max_depth=12, max_features=0.3, n_estimators=100, random_state=SEED)
    random_forest.fit(x_train.values, y_train.values)
    y_rf_test = random_forest.predict(x_test.values)
    print("Random Forest",r2_score(y_test, y_rf_test))

    '''
    votingRegressor = VotingRegressor([('hist',hgBoost),('est', random_forest), ('model_xgb', model_xgb),('GBoost',gBoost),('Adaboost',ada)])
    votingRegressor.fit(x_train.values, y_train.values)
    y_test_predict = votingRegressor.predict(x_test.values)  # dernieres val
    score = r2_score(y_test, y_test_predict)
    '''
