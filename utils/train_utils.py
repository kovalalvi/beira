import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import numpy as np
import os
import matplotlib.pyplot as plt

import geoopt

def make_complex_loss_function(mse_weight = 0., corr_weight = 0., manifold_weight = 0., bound=1):
    
    mse_loss = nn.MSELoss()
    cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-08)
    spd_manifold = geoopt.SymmetricPositiveDefinite()
    
    
    def loss_func(y_hat, y_true):
        """
        y.shape [batch, roi, time]
        """
        
        batch = y_hat.shape[0]
        # L1/L1 loss.
        mse = mse_loss(y_hat[..., bound:-bound], y_true[..., bound:-bound])

        # Correlation 
        y_hat_centre = y_hat - torch.mean(y_hat, -1, keepdim=True)
        y_true_centre = y_true - torch.mean(y_true, -1, keepdim=True)

        corrs = cos_sim( y_hat_centre, y_true_centre)
        corr = torch.mean(corrs)
        corr = torch.nan_to_num(corr, nan=0.0)

        corr_neg = -corr

        # Manifold covariance loss

        cov_matrix_hat = torch.stack([torch.cov(y_) for y_ in y_hat])
        cov_matrix_true = torch.stack([torch.cov(y_) for y_ in y_true])

        man_dists = []
        for batch in range(batch):

            man_dist = spd_manifold.dist(x=cov_matrix_hat[batch] , 
                                         y= cov_matrix_true[batch])
            man_dists.append(man_dist)

        manifold_distance = torch.mean(torch.stack(man_dists))
        manifold_distance = torch.clip(manifold_distance, 0, 100) # values might be very big
        
        manifold_distance = torch.nan_to_num(manifold_distance, nan=0.0)


        total_loss = mse_weight * mse + corr_weight * corr_neg + manifold_weight * manifold_distance

        return total_loss, corr, mse, manifold_distance
    return loss_func




def make_mse_loss():
    criterion = nn.MSELoss()
    cos_metric = nn.CosineSimilarity(dim=-1, eps=1e-08)

    def loss_func(y_hat, y_batch):
        mse_loss = criterion(y_hat, y_batch)
        
        y_hat_centre = y_hat - torch.mean(y_hat, -1, keepdim=True)
        y_true_centre = y_batch - torch.mean(y_batch, -1, keepdim=True)
        
        cos_dist = torch.mean(cos_metric(y_hat_centre, y_true_centre))
        return mse_loss, cos_dist
    return loss_func


def make_mae_loss():
    criterion = nn.L1Loss()
    cos_metric = nn.CosineSimilarity(dim=-1, eps=1e-08)

    def loss_func(y_hat, y_batch):
        mae_loss = criterion(y_hat, y_batch)
        cos_dist = torch.mean(cos_metric(y_hat, y_batch))
        return mae_loss, cos_dist
    return loss_func




def train_step(x_batch, y_batch, model, optimizer, loss_function, scheduler=None):
    
    optimizer.zero_grad()
    y_hat = model(x_batch)
    losses = loss_function(y_hat, y_batch)
    
    losses[0].backward()
    optimizer.step()
    if scheduler is not None: 
        scheduler.step()
    return losses

def save_checkpoint_custom(state, best_model_path):
    """
    Save checkpoint based on state information. Save model.state_dict() weight of models.
    Parameters: 
    state: torch dict weights  
        model.state_dict()
    best_model_path: str
        path to save best model( copy from checkpoint_path)
    """
    
    best_check = os.path.split(best_model_path)[0]
    if not os.path.exists(best_check):
        os.makedirs(best_check)
    
    torch.save(state, best_model_path)
    
    
    
def wanb_train_regression(EPOCHS, model, train_loader, val_loader,
                                 loss_function, train_step, optimizer,
                                 device, raw_test_data, labels, inference_function,
                                 to_many,scheduler = None,
                                 show_info=1, num_losses=10):
    """
    Train model with train_loader.  
    
    
    """
    max_cos_val = -1
    batch_size = train_loader.batch_size
    
    # X_test, y_test = raw_test_data

    print("Starting Training of our model",
          "\nNumber of samples", batch_size*len(train_loader),
          "\nSize of batch:", batch_size, "Number batches", len(train_loader))

    #-----------------------------------------------------------------------#
    
    model = model.to(device)
#     wandb.watch(model, loss_function, log_freq=16)
    
    for epoch in range(1, EPOCHS+1):
        model = model.to(device)

        sum_losses =[ 0 for i in range(num_losses)]
        sum_losses_val =[ 0 for i in range(num_losses)]
        
        # model training
        model.train()
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            
            x_batch = x_batch.to(device, dtype=torch.float)
            y_batch = y_batch.to(device, dtype=torch.float)

            losses = train_step(x_batch, y_batch, model, optimizer, loss_function)
            
            num_losses = len(losses)
            
            for i in range(num_losses):
                sum_losses[i] = sum_losses[i] + losses[i].item()
            
            print('.', sep=' ', end='', flush=True)
        
        # model validation
        model.eval()
        with torch.no_grad():
            for counter_val, (x_batch_val, y_batch_val) in enumerate(val_loader):

                x_batch_val = x_batch_val.to(device, dtype=torch.float)
                y_batch_val = y_batch_val.to(device, dtype=torch.float)

                y_hat_val = model(x_batch_val)
                losses_val = loss_function(y_hat_val, y_batch_val)
                
                for i in range(len(losses_val)):
                    sum_losses_val[i] = sum_losses_val[i] + losses_val[i].item()

            ### add to wanb all losses.
            mean_losses = [loss/(counter+1) for loss in sum_losses]
            mean_losses_val = [loss/(counter_val+1) for loss in sum_losses_val]
            
            if scheduler is not None: 
                scheduler.step(mean_losses_val[1])
            
            for i in range(num_losses):
                wandb.log({"train/loss_" + str(i): mean_losses[i]}, epoch) 
                wandb.log({'val/loss_' + str(i): mean_losses_val[i]}, epoch) 
            
            
            # inference only when cosine distance imroves. 
            if max_cos_val < mean_losses_val[1]:
                max_cos_val = mean_losses_val[1]
                
                fig, fig_bars, corrs = inference_function(model, raw_test_data, 
                                                          labels=labels, 
                                                          device=device, 
                                                          to_many=to_many)

                wandb.log({"val_viz/plot_ts_image": wandb.Image(fig)}, epoch)
                wandb.log({"val_viz/plot_corrs": wandb.Image(fig_bars)}, epoch)   
                wandb.log({"val/corr_mean": np.mean(corrs)}, epoch)
                plt.close(fig)
                plt.close(fig_bars)
                
                # save model in that case. 
                # save weights
                filename = "epoch_{}_val_corr{:.2}.pt".format(epoch, np.mean(corrs))
                filepath_name = os.path.join(wandb.run.dir, filename)
                save_checkpoint_custom(model.state_dict(), filepath_name) 
            

        # Logging and saving
        #-------------------------------------------------------------------------------#
        if epoch % show_info == 0:
            general_out_ = '\nEpoch {} '.format(epoch)
            for i in range(len(sum_losses)):
                tmp_string = 'train loss_{} : {:.3} '.format(i, mean_losses[i])
                tmp_string_val = 'val loss_{} : {:.3} '.format(i, mean_losses_val[i]) 
                general_out_ = general_out_ + tmp_string + tmp_string_val
            print(general_out_)
            
        val_loss = mean_losses_val[0]
        val_acc = mean_losses_val[1]
        
    
    return model