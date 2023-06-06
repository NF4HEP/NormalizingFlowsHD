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
import time
import traceback
from typing import Dict, Any
from tensorflow.python.client import device_lib

sys.path.append('../../../code')
import Bijectors,Distributions,Metrics,MixtureDistributions,Plotters,Trainer,Utils

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

import subprocess
def get_gpu_info():
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"]).decode('utf-8')
        return gpu_info.strip().split('\n')
    except Exception as e:
        print(e)
        return None
gpu_models = get_gpu_info()
if gpu_models:
    training_device = gpu_models[eval(os.environ["CUDA_VISIBLE_DEVICES"])]
    print("Successfully loaded GPU model: {}".format(training_device))
else:
    training_device = 'undetermined'
    print("Failed to load GPU model. Defaulting to 'undetermined'.")

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
        
# Base distribution: first component HalfNormal, others Normal
#class MultiParticleMomentumBijector(tfb.Bijector):
#    def __init__(self, mass, n_particles, validate_args=False, name='MultiParticleMomentumBijector'):
#        super(MultiParticleMomentumBijector, self).__init__(
#            forward_min_event_ndims=1,
#            validate_args=validate_args,
#            name=name)
#        self.mass = mass
#        self.n_particles = n_particles
#
#    def _forward(self, y):
#        y = tf.reshape(y, [-1, self.n_particles, 3])  # reshape y to account for multiple particles
#        p_mag, theta, phi = tf.unstack(y, axis=-1)
#        px = p_mag * tf.sin(phi) * tf.cos(theta)
#        py = p_mag * tf.sin(phi) * tf.sin(theta)
#        pz = p_mag * tf.cos(phi)
#        energy = tf.sqrt(self.mass**2 + p_mag**2)
#        return tf.reshape(tf.stack([energy, px, py, pz], axis=-1), [-1, self.n_particles * 4])  # reshape back to original size
#
#    def _inverse(self, x):
#        x = tf.reshape(x, [-1, self.n_particles, 4])  # reshape x to account for multiple particles
#        energy, px, py, pz = tf.unstack(x, axis=-1)
#        p_mag = tf.sqrt(px**2 + py**2 + pz**2)
#        theta = tf.atan2(py, px)
#        phi = tf.acos(pz / p_mag)
#        return tf.reshape(tf.stack([p_mag, theta, phi], axis=-1), [-1, self.n_particles * 3])  # reshape back to original size
#
#    def _inverse_log_det_jacobian(self, x):
#        return 0.0
#
#    def _forward_log_det_jacobian(self, y):
#        return 0.0
#    
#def generate_multi_particle_4momentum(mass, scale, n_particles):
#    # Define the mixture distribution for phi
#    phi_distribution = tfd.Mixture(
#        cat=tfd.Categorical(probs=[0.5, 0.5]),  # each component has equal weight
#        components=[
#            tfd.Uniform(low=0., high=0.1),
#            tfd.Uniform(low=np.pi-0.1, high=np.pi)
#        ],
#    )
#
#    base_distribution = tfd.Blockwise(
#        [tfd.Gamma(concentration=1.5, rate=1.0 / scale), tfd.Uniform(low=0., high=2. * np.pi), phi_distribution] * n_particles
#    )
#    bijector = MultiParticleMomentumBijector(mass, n_particles)
#    return tfd.TransformedDistribution(base_distribution, bijector)
#
#mass = 0.0  # For example, let's say mass is 1
#scale = 30.0  # For example, let's say scale is 1
#nparticles = 3
#base_dist = generate_multi_particle_4momentum(mass, scale, nparticles)

def generate_base_dist(n_particles,means=None,stds=None):
    vec = []
    if means is None:
        means = np.full(4*n_particles,0.,dtype="float32")
    if stds is None:
        stds = np.full(4*n_particles,1.,dtype="float32")
    for i in range(n_particles):
        vec = vec + [
           #tfd.HalfNormal(scale=stds[4*i]),  # energy, always positive
           tfd.Normal(loc=means[4*i], scale=stds[4*i]),  # energy
           tfd.Normal(loc=means[4*i+1], scale=stds[4*i+1]),  # px
           tfd.Normal(loc=means[4*i+2], scale=stds[4*i+2]),  # py
           tfd.Normal(loc=means[4*i+3], scale=stds[4*i+3])]  # pz
    return tfd.Blockwise(vec)


### Initialize hyperparameters lists ###
batch_size_list=[512]
bijectors_list=['MAFN']
nbijectors_list=[10]
hidden_layers_list=[[128,128,128]]
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
results_dict: Dict[str,Any] = {'run_n': [],'seed_train': [],'seed_test': [], 'ndims':[],'nsamples':[],'correlation':[],'nbijectors':[],'bijector':[],'activation':[],'spline_knots':[],'range_min':[],'eps_regulariser':[],'regulariser':[],'ks_mean':[],'ks_std':[],'ks_list':[],'ad_mean':[],'ad_std':[],'ad_list':[],'wd_mean':[],'wd_std':[],'wd_list':[],'swd_mean':[],'swd_std':[],'swd_list':[],'fn_mean':[],'fn_std':[],'fn_list':[],'hidden_layers':[],'batch_size':[],'epochs_input':[],'epochs_output':[],'batch_size':[],'epochs_input':[],'epochs_output':[],'training_time':[],'prediction_time':[],'total_time':[],'training_device':[]}
hyperparams_dict: Dict[str,Any] = {'run_n': [],'seed_train': [],'seed_test': [], 'ndims':[],'nsamples':[],'correlation':[],'nbijectors':[],'bijector':[],'spline_knots':[],'range_min':[],'hidden_layers':[],'batch_size':[],'activation':[],'eps_regulariser':[],'regulariser':[],'dist_seed':[],'test_seed':[],'training_device':[]} 

### Create 'log' file ####
log_file_name = Utils.create_log_file(mother_output_dir,results_dict)

### Run loop  ###
run_number = 1
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
                            start_global = timer()
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
                                to_run=False
                                #epochs=3
                            #try:
                            if to_run:
                                path_to_weights=path_to_results+'weights/'
                                try:
                                    os.mkdir(path_to_weights)
                                except:
                                    print(path_to_weights+' file exists')
                                print("===========\nStandardizing train/test data for run",run_number,".\n")
                                print("===========\n")
                                start=timer()
                                mean = X_data_train.mean(axis=0)
                                std = X_data_train.std(axis=0)
                                #X_data_train_std = X_data_train / std
                                X_data_train_std = (X_data_train - mean) / std
                                end=timer()
                                train_data_time=end-start
                                print("Train/text data standardized in",train_data_time,"s.\n")       
                                Utils.save_hyperparams(path_to_results,hyperparams_dict,run_number,seed_train,seed_test,ndims,nsamples,corr,bijector_name,nbijectors,spline_knots,range_min,hllabel,batch_size,activation,eps_regulariser,regulariser,seed_dist,seed_test,training_device)
                                local_time = time.localtime()
                                local_time_str = time.strftime('%a, %d %b %Y %H:%M:%S', local_time)
                                print("===========\nRunning",run_number,"/",n_runs,"with hyperparameters:\n",
                                      "timestamp=",local_time_str,"\n",
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
                                      "hidden_layers=",hidden_layers,"\n",
                                      "epocs_input=",epochs,"\n",
                                      "training_device=",training_device,"\n",
                                      "\n===========\n")
                                bijector=Bijectors.ChooseBijector(bijector_name,ndims,spline_knots,nbijectors,range_min,hidden_layers,activation,regulariser,eps_regulariser)
                                Utils.save_bijector_info(bijector,path_to_results)
                                base_dist = generate_base_dist(n_particles=3,means=None,stds=None)
                                nf_dist=tfd.TransformedDistribution(base_dist,bijector)
                                print("NF distribution:",nf_dist,"defined\n")
                                X_data_nf_norm_0 = nf_dist.sample(nsamples).numpy()
                                print("A sample from nf_dist:",X_data_nf_norm_0)
                                print("Its log_prob:",nf_dist.log_prob(X_data_nf_norm_0).numpy())
                                #X_data_nf_0 = X_data_nf_norm_0 * std
                                X_data_nf_0 = X_data_nf_norm_0 * std + mean
                                Plotters.cornerplotter(X_data_test,X_data_nf_0,path_to_results,ndims,norm=True)
                                os.rename(path_to_results+'corner_plot.png',path_to_results+'corner_plot_0.png')
                                start=timer()
                                print("Training model.\n")
                                epochs_input = epochs
                                lr=lr_orig
                                n_displays=1
                                print("Train first sample:",X_data_train_std[0])
                                history, training_time = Trainer.graph_execution(ndims,nf_dist, X_data_train_std,epochs, batch_size, n_displays,path_to_results,load_weights=True,load_weights_path=path_to_weights,lr=lr,patience=patience,min_delta_patience=min_delta_patience,reduce_lr_factor=lr_change,seed=seed_train)
                                t_losses_all=list(history.history['loss'])
                                v_losses_all=list(history.history['val_loss'])
                                lr_all=list(history.history['lr'])
                                epochs_output = len(t_losses_all)
                                training_time=end-start
                                print("Model trained in",training_time,"s.\n")
                                #continue
                                start=timer()
                                try:
                                    print("===========\nComputing predictions\n===========\n")
                                    print("===========\nTrying on GPU\n===========\n")
                                    Utils.reset_random_seeds(seed=seed_test)
                                    #if V is not None:
                                    #    X_data_train = MixtureDistributions.inverse_transform_data(X_data_train,V)
                                    #reload_best
                                    nf_dist,_=Utils.load_model(nf_dist,path_to_results,ndims,lr=.000001)
                                    logprob_nf=nf_dist.log_prob(X_data_test).numpy()
                                    pickle_logprob_nf=open(path_to_results+'logprob_nf.pcl', 'wb')
                                    pickle.dump(logprob_nf, pickle_logprob_nf, protocol=4)
                                    pickle_logprob_nf.close()
                                    #[X_data_test, X_data_nf]=Utils.sample_save(X_data_test,nf_dist,path_to_results,sample_size=ntest_samples,iter_size=10000,seed=seed_test)
                                    [X_data_test, X_data_nf_std]=Utils.sample_save(X_data_test,nf_dist,path_to_results,sample_size=ntest_samples,iter_size=10000,seed=seed_test)
                                    #X_data_nf = X_data_nf_std * std
                                    X_data_nf = X_data_nf_std * std + mean
                                    with open(path_to_results+'nf_sample.npy', 'wb') as f:
                                        np.save(f, X_data_nf, allow_pickle=True)
                                    print("Test first sample:",X_data_test[0])
                                    print("NF first sample:",X_data_nf[0])
                                    start_pred=timer()
                                    ks_mean,ks_std,ks_list,ad_mean,ad_std,ad_list,wd_mean,wd_std,wd_list,swd_mean,swd_std,swd_list,fn_mean,fn_std,fn_list=Metrics.ComputeMetrics(X_data_test,X_data_nf)
                                except:
                                    print("===========\nFailed on GPU, re-trying on CPU\n===========\n")
                                    with tf.device('/device:CPU:0'):
                                        Utils.reset_random_seeds(seed=seed_test)
                                        #if V is not None:
                                        #    X_data_train = MixtureDistributions.inverse_transform_data(X_data_train,V)
                                        #reload_best
                                        nf_dist,_=Utils.load_model(nf_dist,path_to_results,ndims,lr=.000001)
                                        logprob_nf=nf_dist.log_prob(X_data_test).numpy()
                                        pickle_logprob_nf=open(path_to_results+'logprob_nf.pcl', 'wb')
                                        pickle.dump(logprob_nf, pickle_logprob_nf, protocol=4)
                                        pickle_logprob_nf.close()
                                        #[X_data_test, X_data_nf]=Utils.sample_save(X_data_test,nf_dist,path_to_results,sample_size=ntest_samples,iter_size=10000,seed=seed_test)
                                        [X_data_test, X_data_nf_std]=Utils.sample_save(X_data_test,nf_dist,path_to_results,sample_size=ntest_samples,iter_size=10000,seed=seed_test)
                                        #X_data_nf = X_data_nf_std * std
                                        X_data_nf = X_data_nf_std * std + mean
                                        with open(path_to_results+'nf_sample.npy', 'wb') as f:
                                            np.save(f, X_data_nf, allow_pickle=True)
                                        print("Test first sample:",X_data_test[0])
                                        print("NF first sample:",X_data_nf[0])
                                        start_pred=timer()
                                        ks_mean,ks_std,ks_list,ad_mean,ad_std,ad_list,wd_mean,wd_std,wd_list,swd_mean,swd_std,swd_list,fn_mean,fn_std,fn_list=Metrics.ComputeMetrics(X_data_test,X_data_nf)
                                end_pred=timer()
                                prediction_time=end_pred-start_pred
                                try:
                                    Plotters.train_plotter(t_losses_all,v_losses_all,path_to_results)
                                    Plotters.cornerplotter(X_data_test,X_data_nf,path_to_results,ndims,norm=True)
                                    Plotters.marginal_plot(X_data_test,X_data_nf,path_to_results,ndims)
                                    #Plotters.sample_plotter(X_data_test,nf_dist,path_to_results)
                                    print("Plots saved")
                                except:
                                    print("===========\nFailed to plot\n===========\n")
                                end_global=timer()
                                total_time=end_global-start_global
                                results_dict=Utils.ResultsToDict(results_dict,run_number,seed_train,seed_test,ndims,nsamples,corr,bijector_name,nbijectors,activation,spline_knots,range_min,ks_mean,ks_std,ks_list,ad_mean,ad_std,ad_list,wd_mean,wd_std,wd_list,swd_mean,swd_std,swd_list,fn_mean,fn_std,fn_list,hllabel,batch_size,eps_regulariser,regulariser,epochs_input,epochs_output,training_time,prediction_time,total_time,training_device)
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
                                print("Model predictions computed in",prediction_time,"s.\n")
                            else:
                                print("===========\nRun",run_number,"/",n_runs,"already exists. Skipping it.\n")
                                print("===========\n")
                            #except Exception as ex:
                            #    # Get current system exception
                            #    ex_type, ex_value, ex_traceback = sys.exc_info()
                            #    # Extract unformatter stack traces as tuples
                            #    trace_back = traceback.extract_tb(ex_traceback)
                            #    # Format stacktrace
                            #    stack_trace = list()
                            #    for trace in trace_back:
                            #        stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
                            #    if not results_dict_saved:
                            #        results_dict=Utils.ResultsToDict(results_dict,run_number,seed_train,seed_test,ndims,nsamples,corr,bijector_name,nbijectors,activation,spline_knots,range_min,"nan","nan","nan","nan","nan","nan","nan","nan","nan","nan","nan","nan","nan","nan","nan",hllabel,batch_size,eps_regulariser,regulariser,epochs_input,"nan","nan","nan","nan",training_device)
                            #    if not logger_saved:
                            #        Utils.logger(log_file_name,results_dict)
                            #    if not results_current_saved:
                            #        Utils.results_current(path_to_results,results_dict)
                            #    if not details_saved:
                            #        try:
                            #            Utils.save_details_json(hyperparams_dict,results_dict,t_losses_all,v_losses_all,lr_all,path_to_results)
                            #        except:
                            #            Utils.save_details_json(hyperparams_dict,results_dict,None,None,None,path_to_results)
                            #    print("===========\nRun failed\n")
                            #    print("Exception type : %s " % ex_type.__name__)
                            #    print("Exception message : %s" %ex_value)
                            #    print("Stack trace : %s" %stack_trace)
                            #    print("===========\n")   
dict_copy = results_dict.copy()
dict_copy.pop('ks_list')
dict_copy.pop('ad_list')
dict_copy.pop('wd_list')
dict_copy.pop('swd_list')
dict_copy.pop('fn_list')
results_frame=pd.DataFrame(dict_copy)
results_frame.to_csv(mother_output_dir+'results_last_run.txt',index=False)
print("Everything done.")