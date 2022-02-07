
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from tqdm.notebook import tqdm 


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
        for i in tqdm(range(self.max_iter)):
            
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
        for i in tqdm(range(self.max_iter)):
            
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