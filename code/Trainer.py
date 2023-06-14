    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:36:34 2019

@author: reyes-gonzalez
"""
import os
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb= tfp.bijectors
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.stats import epps_singleton_2samp
from scipy.stats import anderson_ksamp
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
from timeit import default_timer as timer
#import Distributions,Metrics#Bijectors
from statistics import mean,median
import math
import pickle
from sklearn.model_selection import train_test_split
import Utils
import json

def loader(nf_dist,path_to_weights):

        print('hello')
          
        n_iter=len(os.listdir(path_to_weights))
        weights_dir=path_to_weights+'iter_'+str(n_iter-1)

        for j in range(len(list(nf_dist.bijector.bijectors))):
                print(j)
                
                name=nf_dist.bijector.bijectors[j].name
                if name=='MAFspline':
                
                    weights_file = open(weights_dir+'/'+name+'_'+str(j)+'.pkl',"rb")
                    weights=pickle.load(weights_file)
                    nf_dist.bijector.bijectors[j].parameters.get('shift_and_log_scale_fn').set_weights(weights)
                                    
                else:
                    continue
        return nf_dist


###training_routine
def graph_execution(ndims,trainable_distribution, X_data,n_epochs, batch_size, n_disp,path_to_results,load_weights=False,load_weights_path=None,lr=.001,patience=30,min_delta_patience=0.001,reduce_lr_factor=0.2,seed=0,stop_on_nan=True):
    Utils.reset_random_seeds(seed)
    #log_file_train=open('train_log.txt','w')
    #log_file_train.write('### '+str(ndims)+' ### \n')
    #log_file_train.close()

    x_ = Input(shape=(ndims,), dtype=tf.float32)
    print(x_)
	
    log_prob_ = trainable_distribution.log_prob(x_)
    print('####### log_prob####')
    print(log_prob_)
    model = Model(x_, log_prob_)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                  loss=lambda _, log_prob: -log_prob)
   
    #print('in trainer')
    #weights=trainable_distribution.bijector.bijectors[0].parameters.get('shift_and_log_scale_fn').get_weights()


    #load_weights=True
    training_time = 0
    train_loss=[]
    val_loss=[]
    if load_weights==True:
    
        try:
           model.load_weights(path_to_results+'/model_checkpoint/weights')
           print('Found and loaded existing weights.')
           #nf_dist=loader(nf_dist,load_weights_path)      
        except:
            print('No weights found. Training from scratch.')
            
        try:
            with open(path_to_results+'/details.json', 'r') as f:
                # Load JSON data from file
                json_file = json.load(f)
                train_loss = json_file['train_loss_history']
                val_loss = json_file['val_loss_history']
                training_time = json_file['time']
                print('Found and loaded existing history.')
        except:
            print('No history found. Generating new history.')

    ns = X_data.shape[0]
    if batch_size is None:
        batch_size = ns


    #earlystopping
    early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=min_delta_patience, patience=patience*1.2, verbose=1,
    mode='auto', baseline=None, restore_best_weights=True
     )
    #reducelronplateau
    reducelronplateau=tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", min_delta=min_delta_patience, patience=patience, verbose=1,
    factor=reduce_lr_factor, mode="auto", cooldown=0, min_lr=lr/1000
     )
    # Display the loss every n_disp epoch
    epoch_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs:
                        print('\n Epoch {}/{}'.format(epoch+1, n_epochs, logs),
                              '\n\t ' + (': {:.4f}, '.join(logs.keys()) + ': {:.4f}').format(*logs.values()))
                                       if epoch % n_disp == 0 else False
    )


    checkpoint=tf.keras.callbacks.ModelCheckpoint(
    path_to_results+'/model_checkpoint/weights',
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
    save_freq="epoch",
    options=None)
                
    StopOnNAN=tf.keras.callbacks.TerminateOnNaN()

    if stop_on_nan==False:
        callbacks=[epoch_callback,early_stopping,reducelronplateau,checkpoint]
    else:
        callbacks=[epoch_callback,early_stopping,reducelronplateau,checkpoint,StopOnNAN]

    start = timer()
    history = model.fit(x=X_data,
                        y=np.zeros((ns, 0), dtype=np.float32),
                        batch_size=batch_size,
                        epochs=n_epochs,
                        validation_split=0.3,
                        shuffle=True,
                        verbose=2,
                        callbacks=callbacks)
    end = timer()
    training_time = training_time + end - start
    
    history.history['loss']=train_loss+history.history['loss']
    history.history['val_loss']=val_loss+history.history['val_loss']
    
    return history, training_time

#######custom training


######this function has to be fixed; do not use
def Callback_ModelCheckPoint(loss,distribution,best_distribution,best_loss):

    if loss<best_loss:
        best_distribution=distribution
        #tf.saved_model.save(best_distribution, "spline_model.pb")
        #with open("spline_model.pcl", "wb" ) as flow_name:
        #    pickle.dump( best_distribution, flow_name)
        
        
        
        
        #nf_dist = pickle.load(open(nf_directory+'/'+bijector_name+"_"+str(ndims)+".pcl", 'rb'))
        best_loss=loss
        print('Found smaller loss')
    else:
        #best_distribution=tf.saved_model.load( "spline_model.pb")
        #best_distribution = pickle.load("spline_model.pcl", 'rb')
        best_sample=best_distribution.sample(5)
        print(distribution.log_prob(best_sample))
        print(best_distribution.log_prob(best_sample))
    return best_distribution,best_loss




def Callback_EarlyStopping(LossList, min_delta=0.0001, patience=40):
    #No early stopping for 2*patience epochs
    if len(LossList)//patience < 2 :
        return False
    #Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
    mean_recent = np.mean(LossList[::-1][:patience]) #last
    #you can use relative or absolute change
    delta_abs = np.abs(mean_recent - mean_previous) #abs change
    delta_abs = np.abs(delta_abs / mean_previous)  # relative change
    if delta_abs < min_delta :
        print("*CB_ES* Loss didn't change much from last %d epochs"%(patience))
        print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
        return True
    else:
        return False
        
        
        
#@tf.function
def eager_execution(distribution,target_train_data,batch_size,epochs):

    print(type(target_train_data.numpy()))
    x_train, x_valid, _, __ = train_test_split(target_train_data.numpy(), target_train_data[:,:1].numpy(), test_size=0.1)
    #x_train=target_train_data
    #x_valid=target_train_data
    #x_train=target_distribution.sample(nsamples)
    x_train=tf.cast(x_train,dtype=tf.float32)
    x_train = tf.data.Dataset.from_tensor_slices(x_train)
    x_train = x_train.batch(batch_size)

    #x_valid = target_distribution.sample(int(nsamples/10))
    x_valid=tf.cast(x_valid,dtype=tf.float32)
    x_valid = tf.data.Dataset.from_tensor_slices(x_valid)
    x_valid = x_valid.batch(batch_size)

    num_epochs = epochs
    opt = tf.keras.optimizers.Adam(learning_rate=.001)
    train_losses = []
    valid_losses = []
    old_distribution=distribution
    
    def loss_function(train_batch):
        return -distribution.log_prob(train_batch)
    
    for train_batch in x_train:
        old_loss=-distribution.log_prob(train_batch)
        break
        
        

    
        
            ####add losses with propershape
    
    

 
    foundNaN=False
    stop=False
    best_distribution=distribution
    best_loss=1000000
    for epoch in range(num_epochs):
        start=timer()
        print("Epoch {}...".format(epoch))
        train_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()
        train_losses_epoch=[]
        for train_batch in x_train:
            with tf.GradientTape() as tape:
                tape.watch(distribution.trainable_variables)
                loss = -distribution.log_prob(train_batch)
     
            if math.isnan(train_loss(loss))==True or np.isinf(train_loss(loss))==True:
                foundNaN=True
                break
            #if train_loss(loss)<0:
            #    stop=True
            #    break
                
            #loss,distribution,old_loss,old_distribution=Callback_AvoidNaN(loss,distribution,old_loss,old_distribution)
            train_loss(loss)
            grads = tape.gradient(loss,distribution.trainable_variables)
            opt.apply_gradients(zip(grads, distribution.trainable_variables))
            
            train_losses_epoch.append(train_loss.result().numpy())
        if foundNaN:
            print('found NaN')
            break
        print('loss: '+str(mean(train_losses_epoch)))
        train_losses.append(train_loss.result().numpy())
        
        #print(distribution.trainable_variables)
      
        # Validation
        for valid_batch in x_valid:
            loss = -distribution.log_prob(valid_batch)
            val_loss(loss)
        valid_losses.append(val_loss.result().numpy())
        
        end=timer()
        early_stop=Callback_EarlyStopping(train_losses, min_delta=0.0001, patience=20)
        #best_distribution,best_loss=Callback_ModelCheckPoint(mean(train_losses_epoch),distribution,best_distribution,best_loss)
        if early_stop:
            break
        #if stop:
        #    print('loss negative')
        #    break
        
        print('time epoch:'+str(end-start) )
    return train_losses,valid_losses


#@tf.function
def eager_execution_old(distribution,target_distribution,nsamples,batch_size,epochs):



    x_train=target_distribution.sample(nsamples)
    x_train = tf.data.Dataset.from_tensor_slices(x_train)
    x_train = x_train.batch(batch_size)

    x_valid=target_distribution.sample(int(nsamples*.2))
    x_valid = tf.data.Dataset.from_tensor_slices(x_valid)
    x_valid = x_valid.batch(batch_size)

    num_epochs = epochs
    opt = tf.keras.optimizers.Adam(learning_rate=.001)
    train_losses = []
    valid_losses = []
    old_distribution=distribution
    
    def loss_function(train_batch):
        return -distribution.log_prob(train_batch)
    
    for train_batch in x_train:
        old_loss=tf.function(loss_function(train_batch))
        break
        
    
    ####add losses with propershape
    foundNaN=False
    stop=False
    best_distribution=distribution
    best_loss=1000
    for epoch in range(num_epochs):
        start=timer()
        print("Epoch {}...".format(epoch))
        train_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()
        train_losses_epoch=[]
        for train_batch in x_train:
            with tf.GradientTape() as tape:
                tape.watch(distribution.trainable_variables)
                loss = tf.function(loss_function(train_batch))
     
            if math.isnan(train_loss(loss))==True or np.isinf(train_loss(loss))==True:
                foundNaN=True
                break
            #if train_loss(loss)<0:
            #    stop=True
            #    break
                
            #loss,distribution,old_loss,old_distribution=Callback_AvoidNaN(loss,distribution,old_loss,old_distribution)
            train_loss(loss)
            grads = tape.gradient(loss,distribution.trainable_variables)
            opt.apply_gradients(zip(grads, distribution.trainable_variables))
            
            train_losses_epoch.append(train_loss.result().numpy())
        if foundNaN:
            print('found NaN')
            break
        print('loss: '+str(mean(train_losses_epoch)))
        train_losses.append(train_loss.result().numpy())
        
        #print(distribution.trainable_variables)
      
        # Validation
        for valid_batch in x_valid:
            loss = -distribution.log_prob(valid_batch)
            val_loss(loss)
        valid_losses.append(val_loss.result().numpy())
        
        end=timer()
        early_stop=Callback_EarlyStopping(train_losses, min_delta=0.0001, patience=20)
        #best_distribution,best_loss=Callback_ModelCheckPoint(mean(train_losses_epoch),distribution,best_distribution,best_loss)
        if early_stop:
            break
        #if stop:
        #    print('loss negative')
        #    break
        
        print('time epoch:'+str(end-start) )
    return train_losses,valid_losses
