
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from tqdm import tqdm 
    
import sklearn
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.kernel_ridge import KernelRidge

def cosine_distance(a, b): 
    """
    x and y should be 1 D vecrors
    Normalize and make dot 
    rescale it from (-1, 1) 
    """
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    cos_distance = 1 - cos_sim
    return cos_distance


class IterativeRidgeRegressor:
    """
    Class of sklearn model which tr
    Only one target for training!
    Input size (n_sample, n_electrode, n_time_freq)
    """
    def __init__(self, 
                 n_electrode=30,
                 n_time_freq=1020, 
                 alphas=[0.1, 10],
                 epsilon=0.01,
                 max_iter=50, 
                 solver ='auto',
                 max_iter_regr = 1000):
        
        self.n_electrode = n_electrode
        self.n_time_freq = n_time_freq
        # He random init
        self.spatial_filter = np.random.randn(n_electrode, 1)*np.sqrt(2/n_electrode)
        self.time_freq_filter = np.random.randn(n_time_freq, 1)*np.sqrt(2/n_time_freq)

        
#         self.spatial_filter = np.ones((n_electrode, 1))
#         self.time_freq_filter = np.ones((n_time_freq, 1))
        
        self.spatial_model = sklearn.linear_model.Ridge(alphas[0], fit_intercept=False, 
                                                       solver = solver, max_iter=max_iter_regr)
        self.time_freq_model = sklearn.linear_model.Ridge(alphas[1], fit_intercept=False, 
                                                         solver = solver, max_iter = max_iter_regr)
        
        self.e = epsilon
        self.max_iter = max_iter
        
    def fit(self, X, y):
        """
        Iterative training of both models
        Parameters: 
        X : 
        
        y: nd.array
            shape N, 1
        """
        for i in range(self.max_iter):
            
            #---------------------------#
            # apply time freq filter and train spatial regressor
            X_spatial = np.dot(X, self.time_freq_filter)
            X_spatial = X_spatial.reshape((-1, self.n_electrode))
            
            # train spatial filter and save weights
            spatial_filter_old = np.copy(self.spatial_filter)
            
            self.spatial_model.fit(X_spatial, y)
            spatial_filter = self.spatial_model.coef_   # ( n_electrode, ) 
            self.spatial_filter = spatial_filter.reshape((self.n_electrode, 1))
            
            # --------------------------#
            # apply spatial filter and train time freq regressor 
            # N, 30, 1020 -> N, 1020, 30 -> N, 1020, 1
            X_time_freq = np.dot(X.transpose(0, 2, 1), self.spatial_filter) 
            X_time_freq = X_time_freq.reshape((-1, self.n_time_freq))
            
            # train time freq regressor and save weights
            time_freq_filter_old = np.copy(self.time_freq_filter)
            
            self.time_freq_model.fit(X_time_freq, y)
            time_freq_filter = self.time_freq_model.coef_   
            self.time_freq_filter = time_freq_filter.reshape((self.n_time_freq, 1))

            #---------------------------#
            # condition for stopping
#             print(spatial_weights_old.shape, self.spatial_filter.shape)
#             spatial_dot = corr_metric(spatial_weights_old, self.spatial_filter)
#             time_freq_dot = corr_metric(time_freq_weights_old, self.time_freq_filter)
            

            spatial_cond = cosine_distance(spatial_filter_old.reshape(-1), spatial_filter.reshape(-1))
            time_freq_cond = cosine_distance(time_freq_filter_old.reshape(-1), time_freq_filter.reshape(-1))

            if (spatial_cond<self.e) and (time_freq_cond < self.e):
                print('Training completes. Num iterations ', i)
                break

    def predict(self, X):
        
        res = np.dot(X, self.time_freq_filter)
        res = np.transpose(res, (0, 2, 1))
        res = np.dot(res, self.spatial_filter)
        
        return res.reshape(-1)
    
    def get_weights(self):
        return sefl.time_freq_filter, self.spatial_filter
    
    
    
    
class IterativeLassoRegressor:
    """
    Class of sklearn model which tr
    Only one target for training!
    Input size (n_sample, n_electrode, n_time_freq)
    """
    def __init__(self, n_electrode=30, n_time_freq=1020, alphas=[0.1, 10], epsilon=0.01, max_iter=50, 
                 solver ='auto', max_iter_regr = 1000):
        
        self.n_electrode = n_electrode
        self.n_time_freq = n_time_freq
        # He random init
        self.spatial_filter = np.random.randn(n_electrode, 1)*np.sqrt(2/n_electrode)
        self.time_freq_filter = np.random.randn(n_time_freq, 1)*np.sqrt(2/n_time_freq)

        
#         self.spatial_filter = np.ones((n_electrode, 1))
#         self.time_freq_filter = np.ones((n_time_freq, 1))
        
        self.spatial_model = sklearn.linear_model.Lasso(alphas[0], fit_intercept=False, 
                                                        max_iter=max_iter_regr)
        self.time_freq_model = sklearn.linear_model.Ridge(alphas[1], fit_intercept=False, 
                                                          max_iter = max_iter_regr)
        
        self.e = epsilon
        self.max_iter = max_iter
        
    def fit(self, X, y):
        """
        Iterative training of both models
        Parameters: 
        X : 
        
        y: nd.array
            shape N, 1
        """
        for i in range(self.max_iter):
            
            #---------------------------#
            # apply time freq filter and train spatial regressor
            X_spatial = np.dot(X, self.time_freq_filter)
            X_spatial = X_spatial.reshape((-1, self.n_electrode))
            
            # train spatial filter and save weights
            spatial_filter_old = np.copy(self.spatial_filter)
            
            self.spatial_model.fit(X_spatial, y)
            spatial_filter = self.spatial_model.coef_   # ( n_electrode, ) 
            self.spatial_filter = spatial_filter.reshape((self.n_electrode, 1))
            
            # --------------------------#
            # apply spatial filter and train time freq regressor 
            # N, 30, 1020 -> N, 1020, 30 -> N, 1020, 1
            X_time_freq = np.dot(X.transpose(0, 2, 1), self.spatial_filter) 
            X_time_freq = X_time_freq.reshape((-1, self.n_time_freq))
            
            # train time freq regressor and save weights
            time_freq_filter_old = np.copy(self.time_freq_filter)
            
            self.time_freq_model.fit(X_time_freq, y)
            time_freq_filter = self.time_freq_model.coef_   
            self.time_freq_filter = time_freq_filter.reshape((self.n_time_freq, 1))

            #---------------------------#
            # condition for stopping
#             print(spatial_weights_old.shape, self.spatial_filter.shape)
#             spatial_dot = corr_metric(spatial_weights_old, self.spatial_filter)
#             time_freq_dot = corr_metric(time_freq_weights_old, self.time_freq_filter)
            

            spatial_cond = cosine_distance(spatial_filter_old.reshape(-1), spatial_filter.reshape(-1))
            time_freq_cond = cosine_distance(time_freq_filter_old.reshape(-1), time_freq_filter.reshape(-1))

            if (spatial_cond<self.e) and (time_freq_cond < self.e):
                print('Training completes. Num iterations ', i)
                break

    def predict(self, X):
        
        res = np.dot(X, self.time_freq_filter)
        res = np.transpose(res, (0, 2, 1))
        res = np.dot(res, self.spatial_filter)
        
        return res.reshape(-1)
    
    def get_weights(self):
        return sefl.time_freq_filter, self.spatial_filter
    
    
    
    
    
    
    





def corr_metric(x, y):
    """
    x and y - 1D vectors
    """
    assert x.shape == y.shape  
    r = np.corrcoef(x, y)[0, 1]
    return r


def train_simple_model(model_creation_func, train_data, val_data):
    """
    model_creation_func - function of creation simple model
    This model shoud predict only one roi.
    for example Ridge regression 
    """
    X_train, y_train = train_data
    X_test, y_test = val_data
    
    models = []
    corr_train = []
    corr_test = []
    y_hats = []
    
    for roi in tqdm(range(y_train.shape[-1])):
        y_train_roi = y_train[:, roi]
        y_test_roi = y_test[:, roi]
        
        model = model_creation_func()
        model.fit(X_train, y_train_roi)
    
        y_hat_train = model.predict(X_train)
        y_hat = model.predict(X_test)
    
        corr_train_tmp = corr_metric(y_hat_train, y_train_roi)
        corr_tmp = corr_metric(y_hat, y_test_roi)

        corr_train.append(corr_train_tmp)
        corr_test.append(corr_tmp)
        y_hats.append(y_hat)
        models.append(model)
    
    return models, np.array(corr_train), np.array(corr_test), np.stack(y_hats)



def get_model_iterative_ridge(params):
    def get_model():
        clf = IterativeRidgeRegressor(**params)
        return clf
    return get_model


def get_Ridge_init_func(alpha = 10):
    print(alpha)
    def get_empty_ridge(alpha = alpha):        
        clf = Ridge(alpha=alpha, fit_intercept=False)
        return clf
    return get_empty_ridge

def get_Lasso_init_func(alpha = 10):
    print(alpha)
    def get_empty_lasso():
        clf = Lasso(fit_intercept=False, alpha=alpha,
                   max_iter=500, selection='random')
        return clf
    return get_empty_lasso