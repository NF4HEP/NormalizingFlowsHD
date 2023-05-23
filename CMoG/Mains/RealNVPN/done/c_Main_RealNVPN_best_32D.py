import os
import sys
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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

def MixtureGaussian(ncomp,ndims,seed=0):
    targ_dist = MixtureDistributions.MixMultiNormal1(ncomp,ndims,seed=seed)
    return targ_dist

### Initialize number of components ###
ncomp=3

### Initialize hyperparameters lists ###
ndims_list=[32]
corr_uncorr_list=["corr"]
regulariser_list=[None]
eps_regularisers=[0]
nsamples_list=[100000]
batch_size_list=[256]
bijectors_list=['RealNVPN']
activation_list=['relu']
nbijectors_list=[5]
hidden_layers_list=[[128,128,128]]
seeds_list = [0, 187, 377, 440, 520, 541, 721, 869, 926, 933]
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
mother_output_dir='../../results/RealNVPN_best/32D/'
try:
    os.mkdir(mother_output_dir)
except:
    print('file exists')

### Initialize dictionaries ###
results_dict: Dict[str,Any] = {'run_n': [],'run_seed': [], 'ndims':[],'nsamples':[],'correlation':[],'nbijectors':[],'bijector':[],'activation':[],'spline_knots':[],'range_min':[],'eps_regulariser':[],'regulariser':[],'ks_mean':[],'ks_std':[],'ad_mean':[],'ad_std':[],'wd_mean':[],'wd_std':[],'swd_mean':[],'swd_std':[],'fn_mean':[],'fn_std':[],'hidden_layers':[],'batch_size':[],'epochs_input':[],'epochs_output':[],'time':[],'batch_size':[],'epochs_input':[],'epochs_output':[],'time':[]}
hyperparams_dict: Dict[str,Any] = {'run_n': [],'run_seed': [], 'ndims':[],'nsamples':[],'correlation':[],'nbijectors':[],'bijector':[],'spline_knots':[],'range_min':[],'hidden_layers':[],'batch_size':[],'activation':[],'eps_regulariser':[],'regulariser':[],'dist_seed':[],'test_seed':[]}

### Create 'log' file ####
log_file_name = Utils.create_log_file(mother_output_dir,results_dict)

### Run loop  ###
run_number = 0
n_runs = len(ndims_list)*len(seeds_list)*len(nsamples_list)*len(corr_uncorr_list)*len(activation_list)*len(eps_regularisers)*len(regulariser_list)*len(bijectors_list)*len(nbijectors_list)*len(spline_knots_list)*len(range_min_list)*len(batch_size_list)*len(hidden_layers_list)
for ndims in ndims_list:
    print("===========\nGenerating test data for ndims=",ndims,".\n")
    print("===========\n")
    start=timer()
    targ_dist=MixtureGaussian(ncomp,ndims,seed=seed_dist)
    X_data_test=targ_dist.sample(ntest_samples,seed=seed_test).numpy()
    end=timer()
    test_data_time=end-start
    print("Test data generated in",test_data_time,"s.\n")
    for seed in seeds_list:
        for nsamples in nsamples_list:
            for activation in activation_list:
                for eps_regulariser in eps_regularisers:
                    for regulariser in regulariser_list:
                        for bijector_name in bijectors_list:
                            for nbijectors in nbijectors_list:
                                for spline_knots in spline_knots_list:
                                    for range_min in range_min_list:
                                        for batch_size in batch_size_list:
                                            for hidden_layers in hidden_layers_list:
                                                for corr in corr_uncorr_list:
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
                                                        to_run=False
                                                    try:
                                                        if to_run:
                                                            path_to_weights=path_to_results+'weights/'
                                                            try:
                                                                os.mkdir(path_to_weights)
                                                            except:
                                                                print(path_to_weights+' file exists')
                                                            succeded=False
                                                            while not succeded:
                                                                Utils.reset_random_seeds(seed)
                                                                print("===========\nGenerating train data for run",run_number,".\n")
                                                                print("===========\n")
                                                                start=timer()
                                                                X_data_train=targ_dist.sample(nsamples,seed=seed).numpy()
                                                                if corr == "corr":
                                                                    V = None
                                                                elif corr == "uncorr":
                                                                    V = MixtureDistributions.rot_matrix(X_data_train)
                                                                    X_data_train = MixtureDistributions.transform_data(X_data_train,V)
                                                                    X_data_test = MixtureDistributions.transform_data(X_data_test,V)
                                                                else:
                                                                    V = None
                                                                end=timer()
                                                                train_data_time=end-start
                                                                print("Train data generated in",train_data_time,"s.\n")       
                                                                hllabel='-'.join(str(e) for e in hidden_layers)
                                                                Utils.save_hyperparams(path_to_results,hyperparams_dict,run_number,seed,ndims,nsamples,corr,bijector_name,nbijectors,spline_knots,range_min,hllabel,batch_size,activation,eps_regulariser,regulariser,seed_dist,seed_test)
                                                                print("===========\nRunning",run_number,"/",n_runs,"with hyperparameters:\n",
                                                                      "ndims=",ndims,"\n",
                                                                      "seed=",seed,"\n",
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
                                                                print("Train first sample:",X_data_train[0])
                                                                history=Trainer.graph_execution(ndims,nf_dist, X_data_train,epochs, batch_size, n_displays,path_to_results,load_weights=True,load_weights_path=path_to_weights,lr=lr,patience=patience,min_delta_patience=min_delta_patience,reduce_lr_factor=lr_change,seed=seed)
                                                                t_losses_all=list(history.history['loss'])
                                                                v_losses_all=list(history.history['val_loss'])
                                                                if len(t_losses_all) > 10:
                                                                    succeded=True
                                                                    end=timer()
                                                                else:
                                                                    print("Training failed: trying again with different seed.")
                                                                    seed = np.random.seed(np.random.randint(1000000))
                                                            epochs_output = len(t_losses_all)
                                                            training_time=end-start
                                                            print("Model trained in",training_time,"s.\n")
                                                            #continue
                                                            start=timer()
                                                            print("===========\nComputing predictions\n===========\n")
                                                            with tf.device('/device:CPU:0'):
                                                                if V is not None:
                                                                    X_data_train = MixtureDistributions.inverse_transform_data(X_data_train,V)
                                                                    X_data_test = MixtureDistributions.inverse_transform_data(X_data_test,V)
                                                                #reload_best
                                                                nf_dist,_=Utils.load_model(nf_dist,path_to_results,ndims,lr=.000001)
                                                                logprob_nf=nf_dist.log_prob(X_data_test).numpy()
                                                                pickle_logprob_nf=open(path_to_results+'logprob_nf.pcl', 'wb')
                                                                pickle.dump(logprob_nf, pickle_logprob_nf, protocol=4)
                                                                pickle_logprob_nf.close()
                                                                [X_data_test, X_data_nf]=Utils.sample_save(X_data_test,nf_dist,path_to_results,sample_size=ntest_samples,rot=V,iter_size=10000,seed=seed)
                                                                print("Test first sample:",X_data_test[0])
                                                                print("NF first sample:",X_data_nf[0])
                                                                ks_mean,ks_std,ad_mean,ad_std,wd_mean,wd_std,swd_mean,swd_std,fn_mean,fn_std=Metrics.ComputeMetrics(X_data_test,X_data_nf)
                                                                results_dict=Utils.ResultsToDict(results_dict,run_number,seed,ndims,nsamples,corr,bijector_name,nbijectors,activation,spline_knots,range_min,ks_mean,ks_std,ad_mean,ad_std,wd_mean,wd_std,swd_mean,swd_std,fn_mean,fn_std,hllabel,batch_size,eps_regulariser,regulariser,epochs_input,epochs_output,training_time)
                                                                results_dict_saved=True
                                                                print("Results dict saved")
                                                                Utils.logger(log_file_name,results_dict)
                                                                logger_saved=True
                                                                print("Logger saved")
                                                                Utils.results_current(path_to_results,results_dict)
                                                                results_current_saved=True
                                                                print("Results saved")
                                                                Utils.save_details_json(hyperparams_dict,results_dict,t_losses_all,v_losses_all,path_to_results)
                                                                details_saved=True
                                                                print("Details saved")
                                                                #Plotters.train_plotter(t_losses_all,v_losses_all,path_to_results)
                                                                #corner_start=timer()
                                                                #Plotters.cornerplotter(X_data_test,X_data_nf,path_to_results,ndims,norm=True)
                                                                #Plotters.marginal_plot(X_data_test,X_data_nf,path_to_results,ndims)
                                                                #Plotters.sample_plotter(X_data_test,nf_dist,path_to_results)
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
                                                            results_dict=Utils.ResultsToDict(results_dict,run_number,seed,ndims,nsamples,corr,bijector_name,nbijectors,activation,spline_knots,range_min,"nan","nan","nan","nan","nan","nan","nan","nan","nan","nan",hllabel,batch_size,eps_regulariser,regulariser,epochs_input,"nan","nan")
                                                        if not logger_saved:
                                                            Utils.logger(log_file_name,results_dict)
                                                        if not results_current_saved:
                                                            Utils.results_current(path_to_results,results_dict)
                                                        if not details_saved:
                                                            try:
                                                                Utils.save_details_json(hyperparams_dict,results_dict,t_losses_all,v_losses_all,path_to_results)
                                                            except:
                                                                Utils.save_details_json(hyperparams_dict,results_dict,None,None,path_to_results)
                                                        print("===========\nRun failed\n")
                                                        print("Exception type : %s " % ex_type.__name__)
                                                        print("Exception message : %s" %ex_value)
                                                        print("Stack trace : %s" %stack_trace)
                                                        print("===========\n")
results_frame=pd.DataFrame(results_dict)
results_frame.to_csv(mother_output_dir+'results_last_run.txt',index=False)
print("Everything done.")