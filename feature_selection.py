import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.feature_selection import VarianceThreshold, GenericUnivariateSelect, SelectFromModel, RFE
from sklearn.feature_selection import f_regression
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from solution import SEED

# More details on https://scikit-learn.org/stable/modules/feature_selection.html

def transform_data(X_train, X_test, sel, y=None):
    # Transform the training set with the selector
    X_train_new = sel.fit_transform(X_train, y=y)
    # Then use the fitted selector to transform test set
    X_test_new = sel.transform(X_test)
    return pd.DataFrame(X_train_new), pd.DataFrame(X_test_new)


def remove_low_var_features(X_train, X_test, threshold=0):
    """Removing features with low variance
    
    Parameters
    ----------
    
    threshold : float, default=0
        Features with a training-set variance lower than this threshold will be removed.
        The default is to keep all features with non-zero variance, i.e. remove the features
        that have the same value in all samples.

    Returns
    -------

    X_train_new, X_test_new : new filtered data
    """
    sel = VarianceThreshold(threshold)
    return transform_data(X_train, X_test, sel)
    

def select_univariate(X_train, X_test, y, score_func=f_regression, mode='percentile', param=0.00001):
    """Selects the best features based on univariate statistical tests.

    Parameters
    ----------

    score_func : Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues).
    For modes ‘percentile’ or ‘kbest’ it can return a single array scores.

    mode : Feature selection mode. Values can be :
    - ‘percentile’ removes all but the k% highest scoring features.
    - ‘k_best’ removes all but the k highest scoring features.
    - ‘fpr’ false positive rate.
    - ‘fdr’ false discovery rate.
    - ‘fwe’ family wise error.

    param : float or int depending on the feature selection mode, default=1e-5
        Parameter of the corresponding mode.

    Returns
    -------

    X_train_new, X_test_new : new filtered data
    """
    sel = GenericUnivariateSelect(score_func=score_func, mode=mode, param=param)
    return transform_data(X_train, X_test, sel, y=y)


def select_from_model(X_train, X_test, y, model='Lasso'):
    """Selects the best features based on univariate statistical tests.

    Parameters
    ----------

    model : Estimator model for regression. Values can be:
    - ‘Lasso‘
    - ‘Ridge‘
    - ‘ElasticNet‘
    - ‘SVR‘

    Returns
    -------

    X_train_new, X_test_new : new filtered data
    """
    if model=='SVR':
        estimator = SVR(kernel="linear")
    elif model=='Ridge':
        estimator = Ridge()
    elif model=='ElasticNet':
        estimator = ElasticNet()
    else:
        estimator = Lasso()
    estimator = estimator.fit(X_train, y)
    sel = SelectFromModel(estimator, prefit=True)
    return pd.DataFrame(sel.transform(X_train)), pd.DataFrame(sel.transform(X_test))


def recursive_feature_elimination(X_train, X_test, y, num_features, estimator):
    ''''
    
    Parameters
    ----------

    num_features : number of features we want to keep after the selection.

    estimator : estimator used for the selection.
        SVR(kernel="linear")
        Lasso()
        Ridge()
        ElasticNet()
    '''
    sel = RFE(estimator, n_features_to_select=num_features, step=0.1)
    x_train = sel.fit_transform(X_train, y)
    x_test = sel.transform(X_test)
    return pd.DataFrame(x_train),pd.DataFrame(x_test)



#Recursive Feature Elimination
def RFE_selector(X_train,X_test,y,num_features):
    estimator = GradientBoostingRegressor(random_state=SEED)
    rfe_selector = RFE(estimator=estimator, n_features_to_select=num_features, step=0.1, verbose=1)
    x_train_rfe = rfe_selector.fit_transform(X_train, y)
    x_test_rfe = rfe_selector.transform(X_test)

    return pd.DataFrame(x_train_rfe),pd.DataFrame(x_test_rfe)

