import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.ensemble import IsolationForest, VotingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.feature_selection import f_regression, chi2
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge


import feature_selection as s

SEED = 400


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

def outliers_KNN(data, ydata):
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
    x_Test =  pd.read_csv("x_test.csv", delimiter=",", index_col='id')

    #formatting
    train_data = pd.merge(left=x_train_origin, right=y_train_origin, how='inner').drop(columns=['id'])
    x_train = x_train_origin.drop(columns=['id'])
    y = y_train_origin['y']

    #Imputation missing values
    x_train, x_Test = miss_val(x_train, x_Test)
    #Remove outliers : Isolation Forest
    x_train, y = outliers_IF(x_train, y)
    #x_train, y = outliers_KNN(x_train, y)

    #Feature Selection :
        # removing all features with variance lower than 0.1 (low var = all samples have the same values)
    #x_train, x_Test = s.remove_low_var_features(x_train, x_Test, 20)
        # example : k_best with 20 best features using f_regression function
    #x_train, x_Test = s.select_univariate(x_train, x_Test, y, mode='k_best', param=200)
        # example : using the lasso model for selection
    #x_train, x_Test = s.select_from_model(x_train, x_Test, y)
        # example : using the lasso model for selection
    #x_train, x_Test = s.recursive_feature_elimination(x_train, x_Test, y, 190,SVR(kernel="linear"))
    x_train, x_Test = s.RFE_selector(x_train, x_Test, y, 180)
    print ("Train shape after selection: {} ".format(x_train.shape))


    #Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x_train, y, test_size=0.33, random_state=SEED)


    #Regressors :
        ###Gradient Boosting Regressor
    gBoost = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1,
                                   max_depth=4, max_features=0.1 ,
                                   min_samples_leaf=15, min_samples_split=10, random_state=SEED)

    gBoost.fit(x_train.values, y_train.values)
    y_gBoost_test = gBoost.predict(x_test.values)
    print("Gradient Boosting", r2_score(y_test, y_gBoost_test))

        ###Random Forest
    random_forest = RandomForestRegressor(max_depth=12, max_features=0.3, n_estimators=100)
    random_forest.fit(x_train.values, y_train.values)
    y_rf_test = random_forest.predict(x_test.values)
    print("Random Forest",r2_score(y_test, y_rf_test))

        ###Cat Boost
    '''
    grid = {
     #"iterations": [100,300],
     "learning_rate": [x * 0.01 for x in range(0, 10)] #,
     #"depth" : [6,7,8,9,10],
     }

    est = CatBoostRegressor()
    logreg_cv=GridSearchCV(est,grid, scoring='r2')
    logreg_cv.fit(x_train.values, y_train.values)
    print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
    print("accuracy :",logreg_cv.best_score_)
    '''

    catboost = CatBoostRegressor()
    catboost.fit(x_train.values, y_train.values)
    y_catboost_test = catboost.predict(x_test.values)
    y_predictions = catboost.predict(x_Test.values)
    y_predictions = np.reshape(y_predictions, y_predictions.shape[0])
    print("Cat Boost",r2_score(y_test, y_catboost_test))


        ###Stochastic Gradient Descent Regression
    sgd = SGDRegressor()
    sgd.fit(x_train.values, y_train.values)
    y_sgd_test = sgd.predict(x_test.values)
    print("SGD",r2_score(y_test, y_sgd_test))

        ###Kernel Ridge
    kernel = KernelRidge()
    kernel.fit(x_train.values, y_train.values)
    y_ker_test = kernel.predict(x_test.values)
    print("Kernel Ridge ",r2_score(y_test, y_ker_test))

        ###Elastic Net
    elasticnet = ElasticNet()
    elasticnet.fit(x_train.values, y_train.values)
    y_elasticnet_test = elasticnet.predict(x_test.values)
    print("Elastic Net ",r2_score(y_test, y_elasticnet_test))

        ###Bayesian Ridge
    bay = BayesianRidge()
    bay.fit(x_train.values, y_train.values)
    y_bay_test = bay.predict(x_test.values)
    print("Bayesian Ridge ",r2_score(y_test, y_bay_test))

        ###Support Vector Machine
    svm = SVR()
    svm.fit(x_train.values, y_train.values)
    y_svm_test = svm.predict(x_test.values)
    print("Support Vector Machine ",r2_score(y_test, y_svm_test))


        ###AdaBoost
    ada = AdaBoostRegressor()
    ada.fit(x_train.values, y_train.values)
    y_ada_train = ada.predict(x_train.values)
    y_ada_test = ada.predict(x_test.values)
    print("Adaboost ",r2_score(y_test, y_ada_test))

        ###MLP Regressor
    nnet = MLPRegressor()
    nnet.fit(x_train.values, y_train.values)
    y_nnet_train = nnet.predict(x_train.values)
    y_nnet_test = nnet.predict(x_test.values)
    print("MLP Regressor ",r2_score(y_test, y_nnet_test))

    #Create submission file
    output = pd.DataFrame()
    output.insert(0, 'y', y_predictions)
    output.index = x_Test.index
    output.index.names = ['id']
    output.to_csv("output")

    '''
    votingRegressor = VotingRegressor([('hist',hgBoost),('est', random_forest), ('model_xgb', model_xgb),('GBoost',gBoost),('Adaboost',ada)])
    votingRegressor.fit(x_train.values, y_train.values)
    y_test_predict = votingRegressor.predict(x_test.values)  # dernieres val
    score = r2_score(y_test, y_test_predict)
    
            ###XGBoost
    xgboost = XGBRegressor()
    xgboost.fit(x_train.values, y_train.values)
    y_xgboost_test = random_forest.predict(x_test.values)
    print("XGBoost",r2_score(y_test, y_xgboost_test))
    
            ###LGBM
    lgbm = LGBMRegressor()
    lgbm.fit(x_train.values, y_train.values)
    y_lgbm_test = random_forest.predict(x_test.values)
    print("LGBM",r2_score(y_test, y_lgbm_test))
    '''
