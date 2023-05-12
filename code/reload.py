import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb= tfp.bijectors
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import LambdaCallback
from scipy import stats
import Distributions,Bijectors
#import Trainer_2 as Trainer
import Metrics as Metrics
from statistics import mean,median
import pickle
from timeit import default_timer as timer
import os
import math
#import CorrelatedGaussians
import Plotters
import MixtureDistributions
#import compare_logprobs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"





def load_hyperparams(path_to_results):

    hyperparams_path=path_to_results+'/hyperparams.txt'
    hyperparams_frame=pd.read_csv(hyperparams_path)
    lastone=int(hyperparams_frame.shape[0]-1)
    print('lastone')
    print(lastone)
    ndims=int(hyperparams_frame['ndims'][lastone])
    nsamples=int(hyperparams_frame['nsamples'][lastone])
    bijector_name=str(hyperparams_frame['bijector'][lastone])
    nbijectors=int(hyperparams_frame['nbijectors'][lastone])
    batch_size=int(hyperparams_frame['batch_size'][lastone])
    spline_knots=int(hyperparams_frame['spline_knots'][lastone])
    range_min=int(hyperparams_frame['range_min'][lastone])
    activation=str(hyperparams_frame['activation'][lastone])
    regulariser=str(hyperparams_frame['regulariser'][lastone])
    eps_regulariser=float(hyperparams_frame['eps_regulariser'][lastone])
    hllabel=str(hyperparams_frame['hidden_layers'][lastone])
    
    
    hidden_layers=hllabel.split('-')
    for i in range(len(hidden_layers)):
        hidden_layers[i]=int(hidden_layers[i])

    return ndims,nsamples,bijector_name,nbijectors,batch_size,spline_knots,range_min,activation,hidden_layers,hllabel,regulariser,eps_regulariser


def MixtureGaussian(ndims):

        if ndims==4:
            targ_dist=MixtureDistributions.MixGauss4()
        
        if ndims==8:
            targ_dist=MixtureDistributions.MixGauss8()

        if ndims==16:
            targ_dist=MixtureDistributions.MixGauss16()

        if ndims==32:
            targ_dist=MixtureDistributions.MixGauss32()
        
        
        if ndims==64:
            targ_dist=MixtureDistributions.MixGauss64()
    
        if ndims==100:
            targ_dist=MixtureDistributions.MixGauss100()
            
        return targ_dist

def ChooseBijector(bijector_name,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser):


    if regulariser=='l1':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
    if regulariser=='l2':
        regulariser=tf.keras.regularizers.l2(eps_regulariser)
    else:
        regulariser=None
    
        

    #if bijector_name=='CsplineN':
    #    rem_dims=int(ndims/2)
    #    bijector=Bijectors.CsplineN(ndims,rem_dims,spline_knots,nbijectors,range_min,hidden_layers,activation)
    
    if bijector_name=='MsplineN':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=Bijectors.MAFNspline(ndims,spline_knots,nbijectors,range_min,hidden_layers,activation,kernel_initializer='glorot_uniform',kernel_regularizer=regulariser)
        


    if bijector_name=='MAFN':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=Bijectors.MAFN(ndims,nbijectors,hidden_layers,activation,kernel_regularizer=regulariser)
        
    if bijector_name=='RealNVPN':
        rem_dims=int(ndims/2)
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=Bijectors.RealNVPN(ndims,rem_dims,nbijectors,hidden_layers,activation,kernel_regularizer=regulariser)

    return bijector
    

def create_flow(bijector_name,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser):
    
    bijector=ChooseBijector(bijector_name,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser)
    base_dist=Distributions.gaussians(ndims)
    nf_dist=tfd.TransformedDistribution(base_dist,bijector)

    return nf_dist

def load_model(nf_dist,path_to_results,ndims,lr=.00001):


    x_ = Input(shape=(ndims,), dtype=tf.float32)
    print(x_)
    print(nf_dist)
    log_prob_ = nf_dist.log_prob(x_)
    print('hello')
    model = Model(x_, log_prob_)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                  loss=lambda _, log_prob: -log_prob)
    model.load_weights(path_to_results+'/model_checkpoint/weights')

    return nf_dist,model
    

        
@tf.function
def save_iter(nf_dist,sample_size,iter_size,n_iters):
    #first iter
    sample_all=nf_dist.sample(iter_size)
    for j in range(1,n_iters):
        if j%100==0:
            print(j/n_iters)
            #print(tf.shape(sample_all))
        sample=nf_dist.sample(iter_size)

        #sample=postprocess_data(sample,preprocess_params)
        sample_all=tf.concat([sample_all,sample],0)
        #if j%1==0:
        #    with open(path_to_results+'/nf_sample_5_'+str(j)+'.npy', 'wb') as f:
        #        np.save(f, sample, allow_pickle=True)
        #tf.keras.backend.clear_session()
    return sample_all



def save_sample(nf_dist,path_to_results,sample_size=100000,iter_size=10000):
    print('saving samples...')
    n_iters=int(sample_size/iter_size)
    
 

    sample_all=save_iter(nf_dist,sample_size,iter_size,n_iters)
    sample_all=sample_all.numpy()
    #print(np.shape(sample_all))
    with open(path_to_results+'/nf_sample.npy', 'wb') as f:
        np.save(f, sample_all, allow_pickle=True)
    print('samples saved')
    return
        
def load_sample(path_to_results):

    nf_sample=np.load(path_to_results+'/nf_sample.npy',allow_pickle=True)
    #nf_sample=np.load(path_to_results+'/sample_nf.pcl',allow_pickle=True)
    
    
    return nf_sample
        



def retrain_model(model,X_data,n_epochs,batch_size,patience=50,min_delta_patience=.00001,n_disp=1):

    ns = X_data.shape[0]
    if batch_size is None:
        batch_size = ns


    #earlystopping
    early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=min_delta_patience, patience=patience, verbose=1,
    mode='auto', baseline=None, restore_best_weights=False
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
    options=None,

                )
                
                
    StopOnNAN=tf.keras.callbacks.TerminateOnNaN()



    history = model.fit(x=X_data,
                        y=np.zeros((ns, 0), dtype=np.float32),
                        batch_size=batch_size,
                        epochs=n_epochs,
                        validation_split=0.3,
                        shuffle=True,
                        verbose=2,
                        callbacks=[epoch_callback,early_stopping,checkpoint,StopOnNAN])
    return history,nf_dist



def ComputeMetrics(X_data_test,nf_dist):

    #kl_divergence=Metrics.KL_divergence(X_data_test,nf_dist,test_log_probs)
    kl_divergence=-1
    ks_test_list=Metrics.KS_test(X_data_test,nf_dist)
    ks_median=median(ks_test_list)
    ks_mean=mean(ks_test_list)
    
    w_distance_list=Metrics.Wasserstein_distance(X_data_test,nf_dist)
    w_distance_median=median(w_distance_list)
    w_distance_mean=median(w_distance_list)
    
    frob_norm,nf_corr,target_corr=Metrics.FrobNorm(X_data_test,nf_dist)
    
    return kl_divergence,ks_median,ks_mean,w_distance_median,w_distance_mean,frob_norm,nf_corr,target_corr

def ResultsToDict(results_dict,ndims,bijector_name,nbijectors,nsamples,activation,spline_knots,range_min,kl_divergence,ks_mean,ks_median,w_distance_median,w_distance_mean,frob_norm,hidden_layers,batch_size,eps_regulariser,regulariser):

            results_dict.get('ndims').append(ndims)
            results_dict.get('bijector').append(bijector_name)
            results_dict.get('nbijectors').append(nbijectors)
            results_dict.get('nsamples').append(nsamples)
            results_dict.get('activation').append(activation)
            results_dict.get('spline_knots').append(spline_knots)
            results_dict.get('range_min').append(range_min)
            results_dict.get('kl_divergence').append(kl_divergence)
            results_dict.get('ks_test_mean').append(ks_mean)
            results_dict.get('ks_test_median').append(ks_median)
            results_dict.get('Wasserstein_median').append(w_distance_median)
            results_dict.get('Wasserstein_mean').append(w_distance_mean)
            results_dict.get('frob_norm').append(frob_norm)
            results_dict.get('time').append(-1)
            results_dict.get('hidden_layers').append(hidden_layers)
            results_dict.get('batch_size').append(batch_size)
            results_dict.get('eps_regulariser').append(eps_regulariser)
            results_dict.get('regulariser').append(regulariser)

            return results_dict

def results_current(path_to_results,results_dict):

    currrent_results_file=open(path_to_results+'results_reload.txt','w')
    header=','.join(list(results_dict.keys()))



    currrent_results_file.write(header)
    currrent_results_file.write('\n')
    
    string_list=[]
    for key in results_dict.keys():
        string_list.append(str(results_dict.get(key)[-1]))
    
    string=','.join(string_list)
    currrent_results_file.write(string)
    currrent_results_file.write('\n')
    
    currrent_results_file.close()


    return
    
    

path='results/test_6/'
results=os.listdir(path)
nsamples_test=10000
nf_sample_exists=True





results_dict={'ndims':[],'nbijectors':[],'bijector':[],'nsamples':[],'activation':[],'spline_knots':[],'range_min':[],'eps_regulariser':[],'regulariser':[],'kl_divergence':[],'ks_test_mean':[],'ks_test_median':[],'Wasserstein_median':[],'Wasserstein_mean':[],'frob_norm':[],'hidden_layers':[],'batch_size':[],'time':[]}







###retrain
epochs=1000
#history,nf_dist=retrain_model(model,X_data_train,epochs,batch_size)
#saver_2(nf_dist,path_to_results,iter)
#saver(nf_dist,path_to_results)
#t_losses=history.history['loss']
#v_losses=history.history['val_loss']


print('save model samples...')

#print(nf_dist.sample(10))
for result in results:

    if 'run_' not in result:
        continue
        
    path_to_results=path+'/'+result+'/'
    print(result)


    files=os.listdir(path_to_results)

    
    ndims,nsamples,bijector_name,nbijectors,batch_size,spline_knots,range_min,activation,hidden_layers,hllabel,regulariser,eps_regulariser=load_hyperparams(path_to_results)




    targ_dist=MixtureGaussian(ndims)
    X_data_test=targ_dist.sample(nsamples_test).numpy()
    
    
    
    with tf.device('/device:CPU:0'):
    
        if nf_sample_exists==False:
        
            nf_dist=create_flow(bijector_name,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser)
            nf_dist,model=load_model(nf_dist,path_to_results,ndims,lr=.00001)
            files=os.listdir(path_to_results)
        
            save_sample(nf_dist,path_to_results,sample_size=nsamples_test,iter_size=400)
        
        
        
        nf_sample=load_sample(path_to_results)


   
        kl_divergence,ks_median,ks_mean,w_distance_median,w_distance_mean,frob_norm,nf_corr,target_corr=ComputeMetrics(X_data_test,nf_sample)
                                    
        results_dict=ResultsToDict(results_dict,ndims,bijector_name,nbijectors,nsamples,activation,spline_knots,range_min,kl_divergence,ks_mean,ks_median,w_distance_median,w_distance_mean,frob_norm,hllabel,batch_size,eps_regulariser,regulariser)
        
        
        results_current(path_to_results,results_dict)
        print('Metrics')
        print(ks_median)
        print(ks_mean)
        print(w_distance_median)
        print(w_distance_mean)

  
        corner_start=timer()
        Plotters.marginal_plot(X_data_test,nf_sample,path_to_results,ndims)
        try:
            Plotters.cornerplotter(X_data_test,nf_sample,path_to_results,ndims,norm=True)
        except:
            continue
        corner_end=timer()
        print(corner_end-corner_start)
        tf.keras.backend.clear_session()
#except:
#    print('no corner plot possible')
