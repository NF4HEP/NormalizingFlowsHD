import os
import sys
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb = tfp.bijectors
import pandas as pd
import pickle
from timeit import default_timer as timer
import traceback
from typing import Dict, Any

sys.path.append('../../../code')
import Bijectors,Distributions,Metrics,MixtureDistributions,Plotters,Trainer,Utils

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
available_gpus = [[x.name, x.physical_device_desc] for x in local_device_protos if x.device_type == 'GPU'] # type: ignore
training_device = str(available_gpus[0])

events_dataset_path = "../../events/events.h5"

print("Loading events dataset...")
start = timer()
with h5py.File(events_dataset_path, 'r') as hdf:
    # Save the datasets into the file with unique names
    try:
        events = np.array(hdf['Z+1j'][:]).astype(np.float32)
    except:
        raise Exception("Z+1j not found in the dataset.")
end = timer()
print("Events dataset loaded in", end - start, "seconds. Shape:", events.shape)
        

### Initialize hyperparameters lists ###
batch_size_list=[512]
bijectors_list=['MAFN']
nbijectors_list=[10]
hidden_layers_list=[[128,128,128],[256,256,256],[512,512,512],[128,128,128,128,128],[256,256,256,256,256],[512,512,512,512,512]]
seeds_list = [0]#, 187, 377, 440, 520, 541, 721, 869, 926, 933]
n_displays=1

### Initialize variables for the neural splines ###
range_min_list=[-5]
spline_knots_list=[8]

### Initialize train hyerparameters ###
ntest_samples=100000
epochs=1000
lr_orig=.001
patience=50
min_delta_patience=.0001
lr_change=.2
seed_dist = 0
seed_test = 0

### Initialize output dir ###
mother_output_dir='../../results/MAFN/Z1j_full/'
try:
    os.mkdir(mother_output_dir)
except:
    print('file exists')

### Initialize dictionaries ###
results_dict: Dict[str,Any] = {'run_n': [],'seed_train': [],'seed_test': [], 'ndims':[],'nsamples':[],'correlation':[],'nbijectors':[],'bijector':[],'activation':[],'spline_knots':[],'range_min':[],'eps_regulariser':[],'regulariser':[],'ks_mean':[],'ks_std':[],'ks_list':[],'ad_mean':[],'ad_std':[],'ad_list':[],'wd_mean':[],'wd_std':[],'wd_list':[],'swd_mean':[],'swd_std':[],'swd_list':[],'fn_mean':[],'fn_std':[],'fn_list':[],'hidden_layers':[],'batch_size':[],'epochs_input':[],'epochs_output':[],'batch_size':[],'epochs_input':[],'epochs_output':[],'training_time':[],'prediction_time':[],'training_device':[]}
hyperparams_dict: Dict[str,Any] = {'run_n': [],'seed_train': [],'seed_test': [], 'ndims':[],'nsamples':[],'correlation':[],'nbijectors':[],'bijector':[],'spline_knots':[],'range_min':[],'hidden_layers':[],'batch_size':[],'activation':[],'eps_regulariser':[],'regulariser':[],'dist_seed':[],'test_seed':[],'training_device':[]} 

### Create 'log' file ####
log_file_name = Utils.create_log_file(mother_output_dir,results_dict)

### Run loop  ###
run_number = 0
corr=None
activation='relu'
nsamples=100000
eps_regulariser=0
regulariser = None
X_data_test = events[-ntest_samples:,:]
X_data_train = events[:nsamples,:]
#X_data_test = np.concatenate([events[-ntest_samples:,i:i+3] for i in range(1, len(events[0]), 4)],axis=1)
#X_data_train = np.concatenate([events[:nsamples,i:i+3] for i in range(1, len(events[0]), 4)],axis=1)
ndims = X_data_test.shape[1]
n_runs = len(seeds_list)*len(bijectors_list)*len(nbijectors_list)*len(spline_knots_list)*len(range_min_list)*len(batch_size_list)*len(hidden_layers_list)
for seed_train in seeds_list:
    for bijector_name in bijectors_list:
        for nbijectors in nbijectors_list:
            for spline_knots in spline_knots_list:
                for range_min in range_min_list:
                    for batch_size in batch_size_list:
                        for hidden_layers in hidden_layers_list:
                            hllabel='-'.join(str(e) for e in hidden_layers)
                            run_number = run_number + 1
                            results_dict_saved=False
                            logger_saved=False
                            results_current_saved=False
                            details_saved=False
                            path_to_results=mother_output_dir+'run_'+str(run_number)+'/'
                            to_run=True
                            try:    
                                os.mkdir(path_to_results)
                            except:
                                print(path_to_results+' file exists')
                                #to_run=False
                                epochs=2
                            try:
                                if to_run:
                                    path_to_weights=path_to_results+'weights/'
                                    try:
                                        os.mkdir(path_to_weights)
                                    except:
                                        print(path_to_weights+' file exists')
                                    succeded=False
                                    mean = X_data_train.mean(axis=0)
                                    std = X_data_train.std(axis=0)
                                    X_data_train_std = (X_data_train - mean) / std
                                    Utils.save_hyperparams(path_to_results,hyperparams_dict,run_number,seed_train,seed_test,ndims,nsamples,corr,bijector_name,nbijectors,spline_knots,range_min,hllabel,batch_size,activation,eps_regulariser,regulariser,seed_dist,seed_test,training_device)
                                    print("===========\nRunning",run_number,"/",n_runs,"with hyperparameters:\n",
                                          "ndims=",ndims,"\n",
                                          "seed_train=",seed_train,"\n",
                                          "nsamples=",nsamples,"\n",
                                          "correlation=",corr,"\n",
                                          "activation=",activation,"\n",
                                          "eps_regulariser=",eps_regulariser,"\n",
                                          "regulariser=",regulariser,"\n",
                                          "bijector=",bijector_name,"\n",
                                          "nbijectors=",nbijectors,"\n",
                                          "spline_knots=",spline_knots,"\n",
                                          "range_min=",range_min,"\n",
                                          "batch_size=",batch_size,"\n",
                                          "hidden_layers=",hidden_layers,
                                          "epocs_input=",epochs,
                                          "\n===========\n")
                                    bijector=Bijectors.ChooseBijector(bijector_name,ndims,spline_knots,nbijectors,range_min,hidden_layers,activation,regulariser,eps_regulariser)
                                    Utils.save_bijector_info(bijector,path_to_results)
                                    base_dist=Distributions.gaussians(ndims)
                                    nf_dist=tfd.TransformedDistribution(base_dist,bijector)
                                    start=timer()
                                    print("Training model.\n")
                                    epochs_input = epochs
                                    lr=lr_orig
                                    n_displays=1
                                    print("Train first sample:",X_data_train_std[0])
                                    history=Trainer.graph_execution(ndims,nf_dist, X_data_train_std,epochs, batch_size, n_displays,path_to_results,load_weights=True,load_weights_path=path_to_weights,lr=lr,patience=patience,min_delta_patience=min_delta_patience,reduce_lr_factor=lr_change,seed=seed_train,stop_on_nan=False)
                                    t_losses_all=list(history.history['loss'])
                                    v_losses_all=list(history.history['val_loss'])
                                    epochs_output = len(t_losses_all)
                                    end=timer()
                                    training_time=end-start
                                    print("Model trained in",training_time,"s.\n")
                                    #continue
                                    start=timer()
                                    try:
                                        print("===========\nComputing predictions\n===========\n")
                                        print("===========\nTrying on GPU\n===========\n")
                                        Utils.reset_random_seeds(seed=seed_test)
                                        print("===========\nGenerating test data for ndims=",ndims,".\n")
                                        print("===========\n")
                                        start=timer()
                                        X_data_test=targ_dist.sample(ntest_samples,seed=seed_test).numpy()
                                        end=timer()
                                        test_data_time=end-start
                                        print("Test data generated in",test_data_time,"s.\n")
                                        #if V is not None:
                                        #    X_data_train = MixtureDistributions.inverse_transform_data(X_data_train,V)
                                        #reload_best
                                        nf_dist,_=Utils.load_model(nf_dist,path_to_results,ndims,lr=.000001)
                                        logprob_nf=nf_dist.log_prob(X_data_test).numpy()
                                        pickle_logprob_nf=open(path_to_results+'logprob_nf.pcl', 'wb')
                                        pickle.dump(logprob_nf, pickle_logprob_nf, protocol=4)
                                        pickle_logprob_nf.close()
                                        [X_data_test, X_data_nf]=Utils.sample_save(X_data_test,nf_dist,path_to_results,sample_size=ntest_samples,iter_size=10000,seed=seed_test)
                                        print("Test first sample:",X_data_test[0])
                                        print("NF first sample:",X_data_nf[0])
                                        start_pred=timer()
                                        ks_mean,ks_std,ks_list,ad_mean,ad_std,ad_list,wd_mean,wd_std,wd_list,swd_mean,swd_std,swd_list,fn_mean,fn_std,fn_list=Metrics.ComputeMetrics(X_data_test,X_data_nf)
                                    except:
                                        print("===========\nFailed on GPU, re-trying on CPU\n===========\n")
                                        with tf.device('/device:CPU:0'):
                                            Utils.reset_random_seeds(seed=seed_test)
                                            print("===========\nGenerating test data for ndims=",ndims,".\n")
                                            print("===========\n")
                                            start=timer()
                                            X_data_test=targ_dist.sample(ntest_samples,seed=seed_test).numpy()
                                            end=timer()
                                            test_data_time=end-start
                                            print("Test data generated in",test_data_time,"s.\n")
                                            #if V is not None:
                                            #    X_data_train = MixtureDistributions.inverse_transform_data(X_data_train,V)
                                            #reload_best
                                            nf_dist,_=Utils.load_model(nf_dist,path_to_results,ndims,lr=.000001)
                                            logprob_nf=nf_dist.log_prob(X_data_test).numpy()
                                            pickle_logprob_nf=open(path_to_results+'logprob_nf.pcl', 'wb')
                                            pickle.dump(logprob_nf, pickle_logprob_nf, protocol=4)
                                            pickle_logprob_nf.close()
                                            [X_data_test, X_data_nf]=Utils.sample_save(X_data_test,nf_dist,path_to_results,sample_size=ntest_samples,iter_size=10000,seed=seed_test)
                                            print("Test first sample:",X_data_test[0])
                                            print("NF first sample:",X_data_nf[0])
                                            start_pred=timer()
                                            ks_mean,ks_std,ks_list,ad_mean,ad_std,ad_list,wd_mean,wd_std,wd_list,swd_mean,swd_std,swd_list,fn_mean,fn_std,fn_list=Metrics.ComputeMetrics(X_data_test,X_data_nf)
                                    end_pred=timer()
                                    prediction_time=end_pred-start_pred
                                    results_dict=Utils.ResultsToDict(results_dict,run_number,seed_train,seed_test,ndims,nsamples,corr,bijector_name,nbijectors,activation,spline_knots,range_min,ks_mean,ks_std,ks_list,ad_mean,ad_std,ad_list,wd_mean,wd_std,wd_list,swd_mean,swd_std,swd_list,fn_mean,fn_std,fn_list,hllabel,batch_size,eps_regulariser,regulariser,epochs_input,epochs_output,training_time,prediction_time,training_device)
                                    results_dict_saved=True
                                    print("Results dict saved")
                                    Utils.logger(log_file_name,results_dict)
                                    logger_saved=True
                                    print("Logger saved")
                                    Utils.results_current(path_to_results,results_dict)
                                    results_current_saved=True
                                    print("Results saved")
                                    Utils.save_details_json(hyperparams_dict,results_dict,t_losses_all,v_losses_all,lr_all,path_to_results)
                                    details_saved=True
                                    print("Details saved")
                                    try:
                                        Plotters.train_plotter(t_losses_all,v_losses_all,path_to_results)
                                        Plotters.cornerplotter(X_data_test,X_data_nf,path_to_results,ndims,norm=True)
                                        Plotters.marginal_plot(X_data_test,X_data_nf,path_to_results,ndims)
                                        #Plotters.sample_plotter(X_data_test,nf_dist,path_to_results)
                                    except:
                                        print("===========\nFailed to plot\n===========\n")
                                    end=timer()
                                    predictions_time=end-start
                                    print("Model predictions computed in",predictions_time,"s.\n")
                                else:
                                    print("===========\nRun",run_number,"/",n_runs,"already exists. Skipping it.\n")
                                    print("===========\n")
                            except Exception as ex:
                                # Get current system exception
                                ex_type, ex_value, ex_traceback = sys.exc_info()
                                # Extract unformatter stack traces as tuples
                                trace_back = traceback.extract_tb(ex_traceback)
                                # Format stacktrace
                                stack_trace = list()
                                for trace in trace_back:
                                    stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
                                if not results_dict_saved:
                                    results_dict=Utils.ResultsToDict(results_dict,run_number,seed_train,seed_test,ndims,nsamples,corr,bijector_name,nbijectors,activation,spline_knots,range_min,"nan","nan","nan","nan","nan","nan","nan","nan","nan","nan","nan","nan","nan","nan","nan",hllabel,batch_size,eps_regulariser,regulariser,epochs_input,"nan","nan","nan","nan")
                                if not logger_saved:
                                    Utils.logger(log_file_name,results_dict)
                                if not results_current_saved:
                                    Utils.results_current(path_to_results,results_dict)
                                if not details_saved:
                                    try:
                                        Utils.save_details_json(hyperparams_dict,results_dict,t_losses_all,v_losses_all,lr_all,path_to_results)
                                    except:
                                        Utils.save_details_json(hyperparams_dict,results_dict,None,None,None,path_to_results)
                                print("===========\nRun failed\n")
                                print("Exception type : %s " % ex_type.__name__)
                                print("Exception message : %s" %ex_value)
                                print("Stack trace : %s" %stack_trace)
                                print("===========\n")   
dict_copy = results_dict.copy()
dict_copy.pop('ks_list')
dict_copy.pop('ad_list')
dict_copy.pop('wd_list')
dict_copy.pop('swd_list')
dict_copy.pop('fn_list')
results_frame=pd.DataFrame(dict_copy)
results_frame.to_csv(mother_output_dir+'results_last_run.txt',index=False)
print("Everything done.")