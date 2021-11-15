import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import DistanceMetric
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
from impyute.imputation.cs import mice, fast_knn
from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler, IterativeImputer
from sklearn.model_selection import GridSearchCV


import feature_selection as s

SEED = 13


#Filling missing values
def miss_val(x_train, x_test):
    #Fill-in nan values with median of the feature
    x_train = x_train.fillna(x_train.median())
    x_test = x_test.fillna(x_test.median())

    return x_train, x_test

def miss_val_with_iterative(x_train,x_test):
    impute_it = IterativeImputer(estimator = GradientBoostingRegressor())
    x_train = pd.DataFrame(impute_it.fit_transform(x_train),columns = x_train.columns)

    impute_it_test = IterativeImputer(estimator = GradientBoostingRegressor())
    x_test = pd.DataFrame(impute_it_test.fit_transform(x_test),columns = x_test.columns)
    return x_train,x_test 

def miss_val_with_knn(x_train,x_test):
    impute_knn = KNNImputer(n_neighbors = 5)
    x_train = pd.DataFrame(impute_knn.fit_transform(x_train),columns = x_train.columns)

    impute_knn_test = KNNImputer(n_neighbors = 5)
    x_test = pd.DataFrame(impute_knn_test.fit_transform(x_test),columns = x_test.columns)
    return x_train,x_test

def miss_val_with_fastknn(x_train,x_test):
    x_train = pd.DataFrame(fast_knn(x_train.values, k=5))

    x_test = pd.DataFrame(fast_knn(x_test.values, k=5))
    return x_train,x_test

def miss_val_with_mice(x_train,x_test):
    x_train = pd.DataFrame(mice(x_train.values))

    x_test = pd.DataFrame(mice(x_test.values))
    return x_train,x_test





'''
def miss_val_with_missForest(x_train,x_test):
    impute_MF = MissForest(max_iter=1, n_estimators=50)
    x_train = pd.DataFrame(impute_MF.fit_transform(x_train),columns = x_train.columns)

    impute_MF = MissForest()
    x_test = pd.DataFrame(impute_MF.fit_transform(x_test),columns = x_train.columns)
    return x_train,x_test
'''
def miss_val_with_fancyinput(x_train,x_test):
    print(x_train)
    X_train_incomplete_normalized = x_train
    impute_fancy = IterativeImputer()
    x_train = pd.DataFrame(impute_fancy.fit_transform(X_train_incomplete_normalized.values),columns = x_train.columns)
    
    X_test_incomplete_normalized = x_test
    impute_fancy_test = IterativeImputer()
    x_test = pd.DataFrame(impute_fancy_test.fit_transform(X_test_incomplete_normalized.values),columns = x_test.columns)
    return x_train,x_test

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
    neigh = NearestNeighbors(n_neighbors=3)
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

def scale(x_train,x_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.fit_transform(x_test)

    return pd.DataFrame(X_train, columns=x_train.columns), pd.DataFrame(X_test, columns = x_test.columns)

def run_model_iter(x_train_start, y_start, x_Test_start, seed_start, seed_stop):
    score = 0
    best_catboost = None
    best_rounded = None
    best_seed = 0
    for i in range(seed_start, seed_stop):
        SEED = i
        #Remove outliers : Isolation Forest
        x_train, y = outliers_IF(x_train_start, y_start)
        #Feature Selection :
        x_train, x_Test = s.RFE_selector(x_train, x_Test_start, y, 50)
        #Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(x_train, y, test_size=0.20, random_state=SEED)

        catboost = CatBoostRegressor(random_seed=SEED)
        catboost.fit(x_train.values, y_train.values)
        y_catboost_test = catboost.predict(x_test.values)
        if r2_score(y_test, y_catboost_test) > score:
            best_seed = i
            best_catboost = y_catboost_test
            best_rounded = np.floor(y_catboost_test) + np.full(np.shape(y_catboost_test), 0.5)
    
    SEED = best_seed
    y_predictions = catboost.predict(x_Test.values) #for output
    #y_predictions = np.floor(y_predictions) + np.full(np.shape(y_predictions), 0.5)
    y_predictions = np.reshape(y_predictions, y_predictions.shape[0]) #for output
    return best_catboost, best_rounded, best_seed, y_predictions, y_test, x_Test

def run_model(x_train, y, x_Test, seed):
    SEED = seed
    #Remove outliers : Isolation Forest
    x_train, y = outliers_IF(x_train, y)
    #x_train, y = outliers_IF(x_train, y)
    #Feature Selection :
    x_train, x_Test = s.RFE_selector(x_train, x_Test, y, 50)
    #Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x_train, y, test_size=0.20, random_state=SEED)

    catboost = CatBoostRegressor(random_seed=200, depth=6, learning_rate=0.05, iterations=1100)
    '''
    parameters = {'depth' : [6],
              'learning_rate' : [0.045,0.05],
              'iterations':[1000]
              }

    grid = GridSearchCV(estimator=catboost, param_grid = parameters, cv = 2, n_jobs=-1)
    result = grid.fit(x_train.values, y_train.values)

    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    y_catboost_test = grid.predict(x_test.values)
    y_predictions = grid.predict(x_Test.values) #for output

    '''
    catboost.fit(x_train.values, y_train.values)
    y_catboost_test = catboost.predict(x_test.values)
    rounded = np.floor(y_catboost_test) + np.full(np.shape(y_catboost_test), 0.5)
    y_predictions = catboost.predict(x_Test.values) #for output
    #y_predictions = np.floor(y_predictions) + np.full(np.shape(y_predictions), 0.5)
    y_predictions = np.reshape(y_predictions, y_predictions.shape[0]) #for output
    return y_catboost_test, rounded, y_predictions, y_test, x_Test

def run_model_submit(x_train, y, x_Test):
    #Remove outliers : Isolation Forest
    x_train, y = outliers_IF(x_train, y)
    #x_train, y = outliers_IF(x_train, y)
    #Feature Selection :
    x_train, x_Test = s.RFE_selector(x_train, x_Test, y, 50)
    #Split the dataset
    #x_train, x_test, y_train, y_test = train_test_split(x_train, y, test_size=0.20, random_state=SEED)

    catboost = CatBoostRegressor(random_seed=200, depth=6, learning_rate=0.05, iterations=9000)
    '''
    parameters = {'depth' : [6],
              'learning_rate' : [0.045,0.05],
              'iterations':[1000]
              }

    grid = GridSearchCV(estimator=catboost, param_grid = parameters, cv = 2, n_jobs=-1)
    result = grid.fit(x_train.values, y_train.values)

    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    y_catboost_test = grid.predict(x_test.values)
    y_predictions = grid.predict(x_Test.values) #for output

    '''
    catboost.fit(x_train.values, y.values)
    #y_catboost_test = catboost.predict(x_test.values)
    #rounded = np.floor(y_catboost_test) + np.full(np.shape(y_catboost_test), 0.5)
    y_predictions = catboost.predict(x_Test.values) #for output
    #y_predictions = np.floor(y_predictions) + np.full(np.shape(y_predictions), 0.5)
    y_predictions = np.reshape(y_predictions, y_predictions.shape[0]) #for output
    return  y_predictions


if __name__ == '__main__':

    x_train_origin =  pd.read_csv("x_train.csv")
    y_train_origin = pd.read_csv("y_train.csv")
    x_Test =  pd.read_csv("x_test.csv", delimiter=",", index_col='id')

    #formatting
    train_data = pd.merge(left=x_train_origin, right=y_train_origin, how='inner').drop(columns=['id'])
    x_train = x_train_origin.drop(columns=['id'])
    y = y_train_origin['y']

    #Imputation missing values
    print("Imputation missing values")
    x_train, x_Test = miss_val_with_knn(x_train, x_Test)
    #scale
    print("scale")
    x_train, x_Test = scale(x_train, x_Test)

    # Set true if you want to test some seeds
    iter = False

    if iter:
        # Seeds that have been tested : 0 -> 10
        catboost, rounded, best_seed, y_predictions, y_test, x_Test = run_model_iter(
            x_train, y, x_Test, 0, 11)
        print("best seed = ", best_seed)
    else:
        #catboost, rounded, y_predictions, y_test, x_Test = run_model(x_train, y, x_Test, SEED)
        y_predictions = run_model_submit(x_train, y, x_Test)

    #print("Cat Boost",r2_score(y_test, catboost))
    #print("with int values", r2_score(y_test, rounded))

    #Create submission file
    output = pd.DataFrame()
    output.insert(0, 'y', y_predictions)
    output.index = x_Test.index
    output.index.names = ['id']
    output.to_csv("output")
