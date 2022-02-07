import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import numpy as np
import os


def make_complex_loss_function(weight_cos_loss=0):
    criterion = nn.MSELoss()
    cos_metric = nn.CosineSimilarity(dim=-1, eps=1e-08)
    
    bound = 2048//8
    
    def loss_func(y_hat, y_batch):
        
        # apply bound for =- 1 sec error in choosing time delay.
        mse_loss = criterion(y_hat[:, :, bound:-bound], y_batch[:, :, bound:-bound])
        
        # cosine sim loss
        cos_sim = torch.mean(cos_metric(y_hat, y_batch))
        cos_loss = -cos_sim
        
        # covariance
        cov_matrix_hat = torch.stack([torch.cov(y_) for y_ in y_hat])
        cov_matrix_hat = torch.triu(cov_matrix_hat, diagonal=0)

        cov_matrix = torch.stack([torch.cov(y_) for y_ in y_batch])
        cov_matrix = torch.triu(cov_matrix, diagonal=0)

        cov_diff_loss = torch.mean((cov_matrix_hat - cov_matrix)**2)
        
        
        all_loss = mse_loss + weight_cos_loss * cos_loss
        return all_loss, cos_sim, mse_loss
    return loss_func




def make_mse_loss():
    criterion = nn.MSELoss()
    cos_metric = nn.CosineSimilarity(dim=-1, eps=1e-08)

    def loss_func(y_hat, y_batch):
        mse_loss = criterion(y_hat, y_batch)
        cos_dist = torch.mean(cos_metric(y_hat, y_batch))
        return mse_loss, cos_dist
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


def wanb_train_regression(EPOCHS, model, train_loader, val_loader,
                                 loss_function, train_step, optimizer,
                                 device, raw_test_data, labels, inference_function,
                                 to_many,
                                 show_info=1, num_losses=10):
    """
    Train model with train_loader.  
    
    
    """
    min_loss = 10000000000
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

                for i in range(num_losses):
                    sum_losses_val[i] = sum_losses_val[i] + losses_val[i].item()

            ### add to wanb all losses.
            mean_losses = [loss/(counter+1) for loss in sum_losses]
            mean_losses_val = [loss/(counter_val+1) for loss in sum_losses_val]
            
            for i in range(num_losses):
                wandb.log({"train/loss_" + str(i): mean_losses[i]}, epoch) 
                wandb.log({'val/loss_' + str(i): mean_losses_val[i]}, epoch) 
            
            
            fig, fig_bars, corrs = inference_function(model, raw_test_data, 
                                                  labels=labels, 
                                                  device=device, to_many=to_many)
            
            wandb.log({"val_viz/plot_ts_image": wandb.Image(fig)}, epoch)
            wandb.log({"val_viz/plot_corrs": wandb.Image(fig_bars)}, epoch)   
            wandb.log({"val/corr_mean": np.mean(corrs)}, epoch)
            


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
        
        
        checkpoint = model.state_dict()
        torch.save(checkpoint, (os.path.join(wandb.run.dir, "model.h5")))
    
    return model