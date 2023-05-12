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
import pandas as pd
import Distributions,Bijectors
import Trainer_5 as Trainer
import Metrics as Metrics
from statistics import mean,median
import pickle
from timeit import default_timer as timer
import os
import math
#import CorrelatedGaussians
import Plotters
import TruncatedDistributions
#import compare_logprobs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def loadData(X_data_train_file,X_data_test_file,logprobs_data_test_file):


        #### import target distribution ######
        
        
        ###pickle train
        print("Importing X_data_train from file",X_data_train_file)
        pickle_train = open(X_data_train_file,'rb')
        start = timer()
        statinfo = os.stat(X_data_train_file)
        X_data_train = pickle.load(pickle_train)
        print(np.shape(X_data_train))
        pickle_train.close()
  
  
        ###pickle test
        print("Importing X_data_test from file",X_data_test_file)
        pickle_test = open(X_data_test_file,'rb')
        start = timer()
        statinfo = os.stat(X_data_test_file)
        X_data_test = pickle.load(pickle_test)
        print(np.shape(X_data_test))
        pickle_test.close()
        
        
        ###pickle test
        print("Importing logprobs_data_test from file",logprobs_data_test_file)
        pickle_logprobs = open(logprobs_data_test_file,'rb')
        start = timer()
        statinfo = os.stat(X_data_test_file)
        logprobs_data_test = pickle.load(pickle_logprobs)
        print(np.shape(logprobs_data_test))
        pickle_logprobs.close()

        end = timer()
        
        print('Files loaded in ',end-start,' seconds.\nFile size is ',statinfo.st_size,'.')
        return X_data_train,X_data_test,logprobs_data_test


        
        
def ChooseBijector(bijector_name,ndims,spline_knots,range_min,hidden_layers,activation,eps_regulariser):

    if bijector_name=='CsplineN':
        rem_dims=int(ndims/2)
        bijector=Bijectors.CsplineN(ndims,rem_dims,spline_knots,nbijectors,range_min,hidden_layers,activation)
    
    if bijector_name=='MsplineN':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=Bijectors.MAFNspline(ndims,spline_knots,nbijectors,range_min,hidden_layers,activation,kernel_initializer='glorot_uniform',kernel_regularizer=regulariser)
        
    if bijector_name=='MsplineNPreprocess':
        regulariser=tf.keras.regularizers.l1(1e-5)
        bijector=Bijectors.MAFNsplinePreprocess(ndims,spline_knots,nbijectors,range_min,hidden_layers,activation,kernel_regularizer=regulariser)

    if bijector_name=='MAFN':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=Bijectors.MAFN(ndims,nbijectors,hidden_layers,activation,kernel_regularizer=regulariser)
        
    if bijector_name=='RealNVPN':
        rem_dims=int(ndims/2)
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=Bijectors.RealNVPN(ndims,rem_dims,nbijectors,hidden_layers,activation,kernel_regularizer=regulariser)

    return bijector

def ComputeMetrics(X_data_test,nf_dist):

    #kl_divergence=Metrics.KL_divergence(X_data_test,nf_dist,test_log_probs)
    kl_divergence=-1
    ks_test_list=Metrics.KS_test(X_data_test,nf_dist)
    ks_median=median(ks_test_list)
    ks_mean=mean(ks_test_list)
    
    ad_test_list=Metrics.AD_test(X_data_test,nf_dist)
    ad_median=median(ad_test_list)
    ad_mean=mean(ad_test_list)
    
    w_distance_list=Metrics.Wasserstein_distance(X_data_test,nf_dist)
    w_distance_median=median(w_distance_list)
    w_distance_mean=median(w_distance_list)
    
    frob_norm,nf_corr,target_corr=Metrics.FrobNorm(X_data_test,nf_dist)
    
    return kl_divergence,ks_median,ks_mean,ad_median,ad_mean,w_distance_median,w_distance_mean,frob_norm,nf_corr,target_corr


def ResultsToDict(results_dict,ndims,bijector_name,nbijectors,nsamples,spline_knots,range_min,kl_divergence,ks_mean,ks_median,ad_mean,ad_median,w_distance_median,w_distance_mean,frob_norm,hidden_layers,batch_size,eps_regulariser):

            results_dict.get('ndims').append(ndims)
            results_dict.get('bijector').append(bijector_name)
            results_dict.get('nbijectors').append(nbijectors)
            results_dict.get('nsamples').append(nsamples)
            results_dict.get('spline_knots').append(spline_knots)
            results_dict.get('range_min').append(range_min)
            results_dict.get('kl_divergence').append(kl_divergence)
            results_dict.get('ks_test_mean').append(ks_mean)
            results_dict.get('ks_test_median').append(ks_median)
            results_dict.get('ad_test_mean').append(ad_mean)
            results_dict.get('ad_test_median').append(ad_median)
            results_dict.get('Wasserstein_median').append(w_distance_median)
            results_dict.get('Wasserstein_mean').append(w_distance_mean)
            results_dict.get('frob_norm').append(frob_norm)
            results_dict.get('time').append(training_time)
            results_dict.get('hidden_layers').append(hidden_layers)
            results_dict.get('batch_size').append(batch_size)
            results_dict.get('eps_regulariser').append(eps_regulariser)

            return results_dict


def ResultsToDict_sc(results_dict_sc,ndims,bijector_name,nbijectors,nsamples,spline_knots,range_min,kl_divergence,ks_mean,ks_median,ad_mean,ad_median,w_distance_median,w_distance_mean,frob_norm,hidden_layers,batch_size,eps_regulariser):

            results_dict_sc.get('ndims').append(ndims)
            results_dict_sc.get('bijector').append(bijector_name)
            results_dict_sc.get('nbijectors').append(nbijectors)
            results_dict_sc.get('nsamples').append(nsamples)
            results_dict_sc.get('spline_knots').append(spline_knots)
            results_dict_sc.get('range_min').append(range_min)
            results_dict_sc.get('kl_divergence').append(kl_divergence)
            results_dict_sc.get('ks_test_mean').append(ks_mean)
            results_dict_sc.get('ks_test_median').append(ks_median)
            results_dict_sc.get('ad_test_mean').append(ad_mean)
            results_dict_sc.get('ad_test_median').append(ad_median)
            results_dict_sc.get('Wasserstein_median').append(w_distance_median)
            results_dict_sc.get('Wasserstein_mean').append(w_distance_mean)
            results_dict_sc.get('frob_norm').append(frob_norm)
            results_dict_sc.get('time').append(training_time)
            results_dict_sc.get('hidden_layers').append(hidden_layers)
            results_dict_sc.get('batch_size').append(batch_size)
            results_dict_sc.get('eps_regulariser').append(eps_regulariser)

            return results_dict_sc

def logger(log_file_name,results_dict):

    log_file=open(log_file_name,'a')
    string_list=[]
    for key in results_dict.keys():
        string_list.append(str(results_dict.get(key)[-1]))
   
    string=','.join(string_list)
    log_file.write(string)
    log_file.write('\n')
    log_file.close()


    return

def results_current(path_to_results,results_dict):

    currrent_results_file=open(path_to_results+'results.txt','w')
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
    
def results_current_sc(path_to_results,results_dict):

    currrent_results_file=open(path_to_results+'results_sc.txt','w')
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
    
def save_hyperparams(path_to_results,hyperparams_dict,ndims,bijector_name,nbijectors,nsamples,spline_knots,range_min,hllabel,batch_size,activation,eps_regulariser):

    
    hyperparams_dict.get('ndims').append(ndims)
    hyperparams_dict.get('bijector').append(bijector_name)
    hyperparams_dict.get('nbijectors').append(nbijectors)
    hyperparams_dict.get('nsamples').append(nsamples)
    hyperparams_dict.get('spline_knots').append(spline_knots)
    hyperparams_dict.get('range_min').append(range_min)
    hyperparams_dict.get('hidden_layers').append(hllabel)
    hyperparams_dict.get('batch_size').append(batch_size)
    hyperparams_dict.get('activation').append(activation)
    hyperparams_dict.get('eps_regulariser').append(eps_regulariser) 
    
    hyperparams_frame=pd.DataFrame(hyperparams_dict)
    hyperparams_frame.to_csv(path_to_results+'/hyperparams.txt',index=False)
    

    return


    




def load_model(nf_dist,path_to_results,ndims,lr=.00001):


    x_ = Input(shape=(ndims,), dtype=tf.float32)
    print(x_)
    print(nf_dist)
    log_prob_ = nf_dist.log_prob(x_)
    print('hello')
    model = Model(x_, log_prob_)
    model.compile(optimizer=tf.optimizers.Adam(lr=lr),
                  loss=lambda _, log_prob: -log_prob)
    model.load_weights(path_to_results+'/model_checkpoint/weights')

    return nf_dist,model


def saver(nf_dist,path_to_weights,iter):
    
                                        
        for j in range(len(list(nf_dist.bijector.bijectors))):
                print(j)
                
                weights_dir=path_to_weights+'iter_'+str(iter)
                try:
                    os.mkdir(weights_dir)
                except:
                    print(weights_dir+' file exists')
                name=nf_dist.bijector.bijectors[j].name
                if name=='MAFspline':
                    weights=nf_dist.bijector.bijectors[j].parameters.get('shift_and_log_scale_fn').get_weights()
                                    
                    weights_file = open(weights_dir+'/'+name+'_'+str(j)+'.pkl', "wb")
                    pickle.dump(weights,weights_file)
                else:
                    continue

        return


def save_bijector_info(bijector,path_to_results):
    
    
    bij_out_file=open(path_to_results+'/bijector_chain.txt','w')
    
    for bij in list(bijector.bijectors):
    
        bij_out_file.write(bij.name)
        bij_out_file.write('\n')
    
    
    
    bij_out_file.close()
    
        
    
    
    return
def SoftClipTransform(nf_dist,ndims,hinge=1e-5):
    if ndims==16:
        max_conditions=[5,4,10,4,4,3,10000,1,5,5,.8,4,4.5,7,0.9525741,0.9890126]
        min_conditions=[-5,-4,0,-2.2,-3,-100,-9999,-1,-4.5,-5,-.8,-2.2,-5.5,-5,4.7425866e-02,1.8393755e-02]
        
    if ndims==8:
        max_conditions=[5,4,10,4,4,3,10000,1]
        min_conditions=[-5,-4,0,-2.2,-3,-100,-9999,-1]
 
    bijector=tfb.SoftClip(low=min_conditions, high=max_conditions,hinge_softness=hinge)

    nf_dist=nf_dist=tfd.TransformedDistribution(nf_dist,bijector)

    return nf_dist


####target dist
trungauss=TruncatedDistributions.some_truncated_gaussians_16()

X_data_test=trungauss.sample(100000).numpy()

print(np.shape(X_data_test)[1])
print(np.min(X_data_test,axis=0))
print(np.max(X_data_test,axis=0))


logprobs_data_test=trungauss.log_prob(X_data_test)





ndims=np.shape(X_data_test)[1]
eps_regularisers=[0]
nsamples_list=[100000]
batch_size_list=[512]
bijectors_list=['RealNVPN','MAFN']

nbijectors_list=[5,8]
hidden_layers_list=[[128,128,128],[512,512,512]]
activation='relu'
n_displays=1
###variables for the neural splines
range_min_list=[-5]
spline_knots_list=[4]

###training vars
epochs=300
lr_orig=.001
#optimizer='adam'
patience=30
min_delta_patience=.0001
train_iters=3
lr_change=.2


###metric vars
nsamples_ks_metrics=1000
nsamples_kl_metrics=10000
nsamples_w_metrics=10000
nsamples_frob_norm=10000
nsamples_corner_plot=10000
#### output_path
mother_output_dir='results/v2_mafn_realnvp_16dims_deep_1/'

try:
    os.mkdir(mother_output_dir)
except:
    print('file exists')


###Initialize dictionary###
results_dict={'ndims':[],'nbijectors':[],'bijector':[],'nsamples':[],'spline_knots':[],'range_min':[],'eps_regulariser':[],'kl_divergence':[],'ks_test_mean':[],'ks_test_median':[],'ad_test_mean':[],'ad_test_median':[],'Wasserstein_median':[],'Wasserstein_mean':[],'frob_norm':[],'hidden_layers':[],'batch_size':[],'time':[]}

results_dict_sc=results_dict
hyperparams_dict={'ndims':[],'nbijectors':[],'bijector':[],'nsamples':[],'spline_knots':[],'range_min':[],'hidden_layers':[],'batch_size':[],'activation':[],'eps_regulariser':[]}
##create 'log' file ####

log_file_name=mother_output_dir+'log_file_eternal.txt'
if os.path.isfile(log_file_name)==False:
    
    log_file=open(log_file_name,'w')
    header=','.join(list(results_dict.keys()))
    print(header)
    log_file.write(header)
    log_file.write('\n')
    log_file.close()


for eps_regulariser in eps_regularisers:
#for ndims in ndims_list:
    for bijector_name in bijectors_list:
        for nbijectors in nbijectors_list:
            for spline_knots in spline_knots_list:
                for range_min in range_min_list:
                    for nsamples in nsamples_list:
                        for batch_size in batch_size_list:
                            for hidden_layers in hidden_layers_list:
                                hllabel='-'.join(str(e) for e in hidden_layers)
                                path_to_results=mother_output_dir+'results_ndims_'+str(ndims)+'_bijector_'+str(bijector_name)+'_nbijectors_'+str(nbijectors)+'_splinekonts_'+str(spline_knots)+'_rangemin_'+str(range_min)+'_nsamples_'+str(nsamples)+'_batchsize_'+str(batch_size)+'_hiddenlayers_'+str(hllabel)+'_eps_regulariser_'+str(eps_regulariser)+'/'
                                
                                
                                try:
                                    os.mkdir(path_to_results)
                                except:
                                    print(path_to_results+' file exists')
                                    
                                    
                                path_to_weights=path_to_results+'/weights/'
                                try:
                                    os.mkdir(path_to_weights)
                                except:
                                    print(path_to_weights+' file exists')
                                    
                                save_hyperparams(path_to_results,hyperparams_dict,ndims,bijector_name,nbijectors,nsamples,spline_knots,range_min,hllabel,batch_size,activation,eps_regulariser)
                                
                                X_data_train=trungauss.sample(nsamples)

 
                                    # with tf.device('/device:GPU:1'):

                                bijector=ChooseBijector(bijector_name,ndims,spline_knots,range_min,hidden_layers,activation,eps_regulariser)
                                    
                                save_bijector_info(bijector,path_to_results)
                                  
                                    
                                base_dist=Distributions.gaussians(ndims)
                                nf_dist=tfd.TransformedDistribution(base_dist,bijector)
                                #print(nf_dist.bijector.bijectors[1].parameters.get('shift_and_log_scale_fn').get_weights())
                                print('hey')
                                '''
                                x_ = Input(shape=(ndims,), dtype=tf.float32)
                                log_prob_ = nf_dist.log_prob(x_)
                                model = Model(x_, log_prob_)
                                print('ho')
                                model.compile(optimizer=tf.optimizers.Adam(),
                                loss=lambda _, log_prob: -log_prob)
                                    
                                nf_dist=loader(nf_dist,path_to_weights)
                                print('yoou')
                                weights=nf_dist.bijector.bijectors[0].parameters.get('shift_and_log_scale_fn').get_weights()
                                print(weights)
                                '''
                                    
                                    
                                    
                                start=timer()
                                #X_data_train=targ_dist.sample(nsamples)
                                #print(target_train_data)
                                #t_losses,v_losses=Trainer.eager_execution(nf_dist,target_train_data,batch_size,epochs)
                                lr=lr_orig
                                for iter in range(train_iters):
                                    if iter>0:
                                       tf.keras.backend.clear_session()             
                                          
                                    n_displays=1
                                    
                                    if iter==0:
                                    
                                        history=Trainer.graph_execution(ndims,nf_dist, X_data_train,epochs, batch_size, n_displays,path_to_results,load_weights=True,load_weights_path=path_to_weights,lr=lr,patience=patience,min_delta_patience=min_delta_patience)
                                    else:
                                        history=Trainer.graph_execution(ndims,nf_dist, X_data_train,epochs, batch_size, n_displays,path_to_results,load_weights=False,load_weights_path=None,lr=lr,patience=patience,min_delta_patience=min_delta_patience)
                                    lr=lr*lr_change
                                    #save weights
                                    n_iter=len(os.listdir(path_to_weights))
                                    #saver(nf_dist,path_to_weights,n_iter)
   
                                      

                                  
                                        
                                        
                                end=timer()
                                training_time=end-start
                                print(training_time)
                                t_losses=history.history['loss']
                                v_losses=history.history['val_loss']
                                #continue
                                with tf.device('/device:CPU:0'):
                                    #reload_best
                                    nf_dist,model=load_model(nf_dist,path_to_results,ndims,lr=.000001)
                                   
                                
                                    logprob_nf=nf_dist.log_prob(X_data_test).numpy()
                                    pickle_logprob_nf=open(path_to_results+'/logprob_nf.pcl', 'wb')
                                    pickle.dump(logprob_nf, pickle_logprob_nf, protocol=4)
                                    pickle_logprob_nf.close()
                            
                                    sample_nf=nf_dist.sample(np.shape(X_data_test)[0]).numpy()
                                    pickle_sample_nf=open(path_to_results+'/sample_nf.pcl', 'wb')
                                    pickle.dump(sample_nf, pickle_sample_nf, protocol=4)
                                    pickle_sample_nf.close()
                            
                            
                                    
                                
                                
                                    
                                
                                
                                    kl_divergence,ks_median,ks_mean,ad_median,ad_mean,w_distance_median,w_distance_mean,frob_norm,nf_corr,target_corr=ComputeMetrics(X_data_test,nf_dist)
                                    
                                    results_dict=ResultsToDict(results_dict,ndims,bijector_name,nbijectors,nsamples,spline_knots,range_min,kl_divergence,ks_mean,ks_median,ad_median,ad_mean,w_distance_median,w_distance_mean,frob_norm,hllabel,batch_size,eps_regulariser)
                            
                                    logger(log_file_name,results_dict)
                                    
                                    '''
                                    try:
                                        compare_logprobs.plot_comparison(logprobs_data_test,logprob_nf,path_to_results)
                                        compare_logprobs.correlation(logprobs_data_test,logprob_nf)
                                    except:
                                        print('problem with logprob comparison. Skipping')
                                    
                                    correlation_dict={'nf_corr':nf_corr,'target_corr':target_corr }
                                    correlation_dict_path = open(path_to_results+'correlation_dict.pkl', "wb")
                                    pickle.dump(correlation_dict,correlation_dict_path)
                                    
                                    '''
                            
                            
                                    Plotters.train_plotter(t_losses,v_losses,path_to_results)
                                  
                         
                                    #Plotters.cornerplotter(X_data_test,nf_dist,path_to_results,ndims)
                                    Plotters.marginal_plot(X_data_test,sample_nf,path_to_results,ndims)
                                    results_current(path_to_results,results_dict)
                            
                            
                            
                                    #Plotters.sample_plotter(X_data_test,nf_dist,path_to_results)
                                    ### now with softclip
                                    
                                    nf_dist=SoftClipTransform(nf_dist,ndims,hinge=1e-5)
                                    
                                    
                                    logprob_nf=nf_dist.log_prob(X_data_test).numpy()
                                    pickle_logprob_nf=open(path_to_results+'/logprob_nf_softclip.pcl', 'wb')
                                    pickle.dump(logprob_nf, pickle_logprob_nf, protocol=4)
                                    pickle_logprob_nf.close()
                            
                                    sample_nf=nf_dist.sample(np.shape(X_data_test)[0]).numpy()
                                    pickle_sample_nf=open(path_to_results+'/sample_nf_softclip.pcl', 'wb')
                                    pickle.dump(sample_nf, pickle_sample_nf, protocol=4)
                                    pickle_sample_nf.close()
                                    
                                    kl_divergence,ks_median,ks_mean,ad_median,ad_mean,w_distance_median,w_distance_mean,frob_norm,nf_corr,target_corr=ComputeMetrics(X_data_test,nf_dist)
                                    
                                    results_dict_sc=ResultsToDict(results_dict,ndims,bijector_name,nbijectors,nsamples,spline_knots,range_min,kl_divergence,ks_mean,ks_median,ad_median,ad_mean,w_distance_median,w_distance_mean,frob_norm,hllabel,batch_size,eps_regulariser)
                                    Plotters.marginal_plot_sc(X_data_test,sample_nf,path_to_results,ndims)
                                    results_current_sc(path_to_results,results_dict_sc)



                                    


results_frame=pd.DataFrame(results_dict)
results_frame.to_csv(mother_output_dir+'/results_last_run.txt',index=False)

results_frame_sc=pd.DataFrame(results_dict_sc)
results_frame_sc.to_csv(mother_output_dir+'/results_last_run_sc.txt',index=False)
        
