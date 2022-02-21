import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import torch 

from tqdm.notebook import tqdm 


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    # max_pixel = np.max([np.max(original), np.max(compressed)])
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def corr_metric(x, y):
    """
    x and y - 1D vectors
    """
    assert x.shape == y.shape,  f'{x.shape} and {y.shape}'
    r = np.corrcoef(x, y)[0, 1]
    return r


# def make_visualization(y_prediction, y_test, labels):
#     """
#     y_prediction - time, roi
#     y_test = time, roi
#     labels - list [name of roi]  
#     """

#     fig, ax = plt.subplots(7, 3, figsize = (15, 12), sharex=True, sharey=True)
#     for roi in tqdm(range(21)):
#         y_hat = y_prediction[:, roi]
#         y_test_roi = y_test[:, roi]
#         corr_tmp = corr_metric(y_hat, y_test_roi)
#         axi = ax.flat[roi]
#         axi.plot(y_hat, label= 'prediction')
#         axi.plot(y_test_roi, label = 'true')
#         axi.set_title("RoI {}_corr {:.2f}".format(labels[roi], corr_tmp))
        
#     return fig 
def corr_results_plot(labels, corrs):
    """
    Return well looking plot by grouping the same RoI. 
    Left and right aggregation 
    Corrrectnees - True if all name is the same. 
    """
    
    labels_roi = labels
    df = pd.DataFrame({'roi_name': labels_roi,
                       'corrs': corrs})


    left_idx = df['roi_name'].str.contains("Left")
    right_idx = df['roi_name'].str.contains("Right")
    not_left_right_idx = ~left_idx &  ~right_idx 

    # left_df = df[left_idx].append(df[not_left_right_idx])
    
    left_df = pd.concat([df[left_idx], df[not_left_right_idx]])
    left_df = left_df.reset_index()

    right_df = pd.concat([df[right_idx], df[not_left_right_idx]])
    right_df = right_df.reset_index()


    # create data
    correctness = False
    correctness =(left_df['roi_name'].str.replace('Left', '') == right_df['roi_name'].str.replace('Right', '')).all()  

    fig_bars = plt.figure(figsize=(8, 8), dpi = 125)
    x = np.arange(len(left_df['corrs']))
    y1 = np.random.rand(len(x))
    y2 = np.random.rand(len(x))
    width = 0.4

    # plot data in grouped manner of bar type
    plt.bar(x-0.2, left_df['corrs'], width, label = 'left')
    plt.bar(x+0.2, right_df['corrs'], width, label = 'right')
    plt.xticks(x, left_df['roi_name'].str.replace('Left', ''), 
              rotation=90)
    plt.ylim(-1, 1)
    plt.title('Test corrs. Correctnees_visual = {}. Average corr {:.2f}'.format(str(correctness),
                                                                                np.mean(corrs)))
    plt.legend()
    
    return fig_bars


def calculate_corrs(y_prediction, y_test):
    """
    Calculate correlation metrics for each roi between y_pred and y_test.
    Visaulize via n_roi plots 
    
    ------
    Input:
        y_prediction - roi, time, 
        y_test - (roi, time) 
        labels - list name of roi
        
    Output
        corrs - list of correaltions.
    """
    corrs = []
    for roi in range(y_test.shape[0]):
        y_hat = y_prediction[roi]
        y_test_roi = y_test[roi]
        corr_tmp = corr_metric(y_hat, y_test_roi)
        corrs.append(corr_tmp)
    return corrs
def make_visualization(y_prediction, y_test, labels):
    """
    Calculate correlation metrics for each roi between y_pred and y_test.
    Visaulize via n_roi plots 
    
    ------
    Input:
        y_prediction - roi, time, 
        
        y_test - (roi, time) 
        
        labels - list name of roi
        
    Output
        fig - bar plot
        corrs - list of correaltions.
    """
    fig, ax = plt.subplots(7, 3, figsize = (10, 15), sharex=True, sharey=True)
    corrs = []
    for roi in range(len(labels)):
        y_hat = y_prediction[roi]
        y_test_roi = y_test[roi]
        corr_tmp = corr_metric(y_hat, y_test_roi)
        corrs.append(corr_tmp)
        axi = ax.flat[roi]
        axi.plot(y_hat, label= 'prediction')
        axi.plot(y_test_roi, label = 'true')

        axi.set_title("{}_corr {:.2f}".format(labels[roi], corr_tmp))
    return fig, corrs



def make_inference_seq_2_seq(model, dataset, device = 'cpu'):
    """
    Make inference for model many2many. So we take sequency of input data and get sequence of output.
    Then we compare prediction and real output. 
    Model should be fully convolution for working with any input data. 
    
    To do
        - add opportunity to predict with sliding window.( overlap or not) 
        - make visualization work only with 21 roi. 
    ------
    Input: 
    
    dataset - (x, y) 
        x - (ch, freq, Time) 
        y - (n_roi, Time)
    
    model - torch model.
        model input  - [batch, channels, freq, time]
        model output  - [batch, n_roi, time]
        
    Output:
        y_hat
        
    """
    
    
    x, y = dataset
    
    with torch.no_grad():
        
        model = model.to(device)
        model.eval()
        
        bound = x.shape[-1]//1024*1024
        
        X_test = x[..., :bound]
        y_test = y[..., :bound]


        x_batch = torch.from_numpy(X_test).float().to(device)
        x_batch = torch.unsqueeze(x_batch, 0)

        y_hat = model(x_batch)[0]
        y_hat = y_hat.to('cpu').detach().numpy()
    
    return y_hat, y_test
        
    
    

    
    
def make_inference_seq_2_one(model, dataset, device = 'cpu'):
    """
    
    
    dataset - (x, y) 
        where x and y just preproc data. 
        
    x, y shapes. It is numpy test dataset. On which we wanna get metrics.
        (ch, freq, Time) - (21, Time)
    
    model input  - [batch, channel, freq,  time]
    

    labels - list of RoI's names.
    
    """
    x, y = dataset
    
    lenght = x.shape[-1]
    
    window_size = model.window_size
    max_start = lenght-window_size
    
    y_test = y[..., window_size-1:]
    y_hats = []
    
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        
        for start in range(0, max_start+1):
            
            end = start+window_size
            X_test = x[..., start:end]

            x_batch = torch.from_numpy(X_test).float().to(device)
            x_batch = torch.unsqueeze(x_batch, 0)

            y_hat = model(x_batch)[0]
            y_hat = y_hat.to('cpu').detach().numpy()
            y_hats.append(y_hat)

    y_hats = np.stack(y_hats, axis =-1)
    
    
    
    return y_hats, y_test





def model_inference_function(model, dataset, 
                             labels, 
                             device='cuda',
                             to_many=False):
    """
    Make inference on test data. 
    Takes model input and output. 
    
    Return 2 graphs. 
    fig - timeserias.
    fig_bars - bar plots with correlations.
    corrs - correlations.
    """
    if to_many:
        y_hats, y_test = make_inference_seq_2_seq(model, dataset, device=device)
    else:
        y_hats, y_test = make_inference_seq_2_one(model, dataset, device=device)
    
    
    fig, corrs = make_visualization(y_hats, y_test, labels = labels)
    fig_bars = corr_results_plot(labels=labels, corrs=corrs)
    
    return fig, fig_bars, corrs





# this is approach to calculate metric on subsampel and then average it. so we can obtain deviation.
def calculate_metric(y_hat, y_true, func_metric_1, func_metric_2, crop_lenght = None ):
    """
    Inputs:
        func_metric_1, func_metric_2 
        are fucnction that calculate some regression metrics for two time serias
    numpy array
    y_hat, y_true have shapes [time_lenght, feature]
    calculate correlation and mse metrics.
    crop on parts and make average estimation.
    if crop_percent=1 we obtain one value for each feature
    ---------------------------------------
    Returns:
    metrics with shape metric for each feature and for each crop [n_feature, n_crops]
    """
    time_lenght = y_hat.shape[0]
    if crop_lenght is None:
        crop_lenght = time_lenght
    
    n_feature = y_hat.shape[-1]
    metrics_1 = [[] for i in range(n_feature)] 
    metrics_2 = [[] for i in range(n_feature)]
    
#     crop_lenght = int(time_lenght * crop_percent)
    n_crops = time_lenght//crop_lenght
    
    # we divide time serias on crop wiht certain lenght. It allows estimate scatter 
    # and understand stability of algorith.
    for i in range(n_crops):
        y_hat_crop = y_hat[i*crop_lenght: (i+1)*crop_lenght]
        y_true_crop = y_true[i*crop_lenght: (i+1)*crop_lenght]
        
        for n_feature in range(y_hat_crop.shape[-1]):
            
            metric_1 = func_metric_1(y_hat_crop[:, n_feature], y_true_crop[:, n_feature])
            metric_2 = func_metric_2(y_hat_crop[:, n_feature], y_true_crop[:, n_feature])
#             print(metric_1, metric_2)

            metrics_1[n_feature].append(metric_1)
            metrics_2[n_feature].append(metric_2)
            
    
    return np.concatenate(metrics_1), np.concatenate(metrics_2) 
    