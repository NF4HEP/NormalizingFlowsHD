visible_devices = [0]
print("Visible devices:", visible_devices)
import datetime
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing os...")
import os
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing re...")
import re
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing timer from timeit...")
from timeit import default_timer as timer
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Setting env variables for tf import (only device", visible_devices, "will be available)...")
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in visible_devices])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing numpy...")
import numpy as np
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing matplotlib...")
from matplotlib import pyplot as plt
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing sys...")
import sys
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing ast...")
import ast
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing h5py...")
import h5py
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing importlib.util...")
import importlib.util
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing json...")
import json
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing io.StringIO...")
from io import StringIO
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing pandas...")
import pandas as pd
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing random...")
import random
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing shutil...")
import shutil
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing scipy utils...")
from scipy.stats import norm, chi2, kstwo, kstwobign, ks_2samp
from scipy.special import kolmogorov
from scipy.optimize import minimize, curve_fit, root, bisect
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing subprocess...")
import subprocess
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing tensorflow...")
import tensorflow as tf
print("Tensorflow version:", tf.__version__)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing tensorflow_probability...")
import tensorflow_probability as tfp
tfd = tfp.distributions
print("Tensorflow probability version:", tfp.__version__)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing timeit...")
from timeit import default_timer as timer
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing tqdm...")
from tqdm import tqdm
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing typing...")
from typing import List, Tuple, Dict, Callable, Union, Optional, Any, Type
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Setting tf configs...")
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu_device in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu_device, True)

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing custom module...")
sys.path.insert(0,'../code/')

import Bijectors,Distributions,Metrics,MixtureDistributions,Plotters,Trainer,Utils

#def import_module_from_path(module_name, path):
#    spec = importlib.util.spec_from_file_location(module_name, path)
#    module = importlib.util.module_from_spec(spec)
#    spec.loader.exec_module(module)
#    return module
#
#MixtureDistributions = import_module_from_path('MixtureDistributions', '../code/MixtureDistributions.py')
#Metrics = import_module_from_path('Metrics', '../code/Metrics.py')
#Utils = import_module_from_path('Utils', '../code/Utils.py')

paper_fig_dir = "../../../NormalizingFlows/papers/NFHD/figures/"

def get_gpu_info():
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"]).decode('utf-8')
        return gpu_info.strip().split('\n')
    except Exception as e:
        print(e)
        return None
gpu_models = get_gpu_info()
if gpu_models:
    training_device = gpu_models[0]#gpu_models[eval(os.environ["CUDA_VISIBLE_DEVICES"])]
    print("Successfully loaded GPU model: {}".format(training_device))
else:
    training_device = 'undetermined'
    print("Failed to load GPU model. Defaulting to 'undetermined'.")
    
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "All done.")

def import_csv_with_lists(run):
    # Preprocess the CSV file
    with open(run, 'r') as f:
        content = f.read()

    # Replace commas inside brackets with semicolon
    content_new = re.sub(r'\[(.*?)]', lambda x: x.group().replace(',', ';'), content)

    # Use StringIO to read the modified CSV content directly into a DataFrame
    df = pd.read_csv(StringIO(content_new))

    # Identify the list of columns that should be converted back to list from string
    #list_columns = ['ks_mean', 'ks_std', 'ad_mean', 'ad_std', 'wd_mean', 'wd_std', 'swd_mean', 'swd_std', 'fn_mean', 'fn_std']
    list_columns = [col for col in df.columns if any(';' in str(x) for x in df[col])]

    # Convert strings back to lists
    for col in list_columns:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x.replace(';', ',')))
    return df

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
def load_model(nf_dist,path_to_results,ndims,lr=.00001,dtype=None):
    """
    Function that loads a model by recreating it, recompiling it and loading checkpointed weights.
    """
    if dtype is None:
        dtype = tf.float32
    x_ = Input(shape=(ndims,), dtype=dtype)
    log_prob_ = nf_dist.log_prob(x_)
    model = Model(x_, log_prob_)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
                  loss=lambda _, log_prob: -log_prob)
    model.load_weights(path_to_results+'model_checkpoint/weights')
    
    return nf_dist,model
    
@tf.function
def generate_samples(dist, batch_size, seed):
    return dist.sample(batch_size, seed=seed)
    
def generate_and_clean_data(dist, n_samples, batch_size, seed, gpu = True):
    X_data = []
    total_samples = 0
    
    if gpu:
        device_name = tf.test.gpu_device_name()
    else:
        device_name = '/device:CPU:0'

    while total_samples < n_samples:
        try:
            with tf.device(device_name):  # Adjust as necessary
                batch = generate_samples(dist, batch_size, seed)

                # Find finite values
                finite_indices = tf.reduce_all(tf.math.is_finite(batch), axis=1)

                # Warn the user if there are any non-finite values
                n_nonfinite = tf.reduce_sum(tf.cast(~finite_indices, tf.int32))
                if n_nonfinite > 0:
                    print(f"Warning: Removed {n_nonfinite} non-finite values from the batch")

                # Select only the finite values
                finite_batch = batch.numpy()[finite_indices.numpy()].astype("float32")

                X_data.append(finite_batch)
                total_samples += len(finite_batch)
        except (RuntimeError, tf.errors.ResourceExhaustedError):
            # If a RuntimeError or a ResourceExhaustedError occurs (possibly due to OOM), halve the batch size
            batch_size = batch_size // 2
            print("Warning: Batch size too large. Halving batch size to {}".format(batch_size),"and retrying.")
            if batch_size == 0:
                raise RuntimeError("Batch size is zero. Unable to generate samples.")
            tf.keras.backend.clear_session()  # Clear the TensorFlow session to free up GPU memory

    return np.concatenate(X_data, axis=0)[:n_samples]


    return np.concatenate(X_data, axis=0)[:n_samples]

def recompute_metrics(run, n_iter = 10, ntest_samples = 100000, batch_size_gen = "auto", n_slices_factor = 2, gpu = True, verbose = True):
    if gpu:
        device_name = tf.test.gpu_device_name()
    else:
        device_name = '/device:CPU:0'
    path_to_results = run.replace("results.txt","")
    results_source_file = run
    details_source_file = run.replace('results.txt','details.json')
    results_original_file = run.replace('results.txt','results_original.txt')
    details_original_file = run.replace('results.txt','details_original.json')
    results_bkp_file = run.replace('results.txt','results_bkp.txt')
    details_bkp_file = run.replace('results.txt','details_bkp.json')
    if os.path.exists(results_original_file):
        print(f"Original file {results_original_file} already exists.")
    else:
        shutil.copyfile(results_source_file, results_original_file)
    if os.path.exists(details_original_file):
        print(f"Original file {details_original_file} already exists.")
    else:
        shutil.copyfile(details_source_file, details_original_file)
    shutil.copyfile(results_source_file, results_bkp_file)
    shutil.copyfile(details_source_file, details_bkp_file)
    try:
        print("\n------------------------------------")
        print("Recomputing metrics for run",run.replace("results.txt",""),".")
        start_all=timer()
        start=timer()
        tmp = import_csv_with_lists(run)
        with open(details_source_file, 'r') as f:
            tmp_json = json.load(f)
        run_n=int(tmp["run_n"])
        seed_train=int(tmp["seed_train"])
        seed_test=int(tmp["seed_train"])
        ndims=int(tmp["ndims"])
        nsamples=int(tmp["nsamples"])
        correlation=None
        nbijectors=int(tmp["nbijectors"])
        bijector=str(tmp["bijector"][0])
        activation=str(tmp["activation"][0])
        spline_knots=int(tmp["spline_knots"])
        range_min=int(tmp["range_min"])
        eps_regulariser=tmp["eps_regulariser"][0]
        regulariser=None
        hidden_layers = str(tmp["hidden_layers"][0])
        hidden_layers_input = [int(i) for i in hidden_layers.split('-')]
        batch_size=int(tmp["batch_size"])
        epochs_input=int(tmp["epochs_input"])
        epochs_output=int(tmp["epochs_output"])
        training_time=tmp["training_time"][0]
        prediction_time=tmp["prediction_time"][0]    
        total_time=tmp["total_time"][0]
        training_device=str(tmp["training_device"][0])
        dist_seed=int(tmp_json["dist_seed"])
        test_seed=int(tmp_json["test_seed"])
        train_loss_history=tmp_json["train_loss_history"]
        val_loss_history=tmp_json["val_loss_history"]
        lr_histoty=tmp_json["lr_history"]
        ncomp=3
        seed_dist = dist_seed
        seed_test = test_seed

        with tf.device(device_name):
        #with tf.device('/GPU:0'):
            # Rebuid the model
            targ_dist = MixtureDistributions.MixMultiNormal1(n_components = ncomp,
                                                             n_dimensions = ndims,
                                                             seed = seed_dist,
                                                             dtype = tf.float32)
            bijector_obj = Bijectors.ChooseBijector(bijector_name = bijector,
                                                    ndims = ndims,
                                                    spline_knots = spline_knots,
                                                    nbijectors = nbijectors,
                                                    range_min = range_min,
                                                    hidden_layers = hidden_layers_input,
                                                    activation = activation,
                                                    regulariser = regulariser,
                                                    eps_regulariser = eps_regulariser,
                                                    perm_style='bi-partition',
                                                    shuffle='Noshuffle')
            base_dist = Distributions.gaussians(ndims = ndims,
                                                dtype = targ_dist.dtype)
            nf_dist = tfd.TransformedDistribution(distribution = base_dist, 
                                                  bijector = bijector_obj)
            print("Loading model\n",run,"\n...")
            nf_dist, _ = load_model(nf_dist = nf_dist,
                                    path_to_results = path_to_results,
                                    ndims = ndims,
                                    lr = .000001,
                                    dtype = targ_dist.dtype)
            end = timer()
            print("Model\n",run,"\nsuccesfully loaded in",end-start,"s.")

            print("Generating samples...")
            start=timer()

            #with tf.device('/GPU:1'):
            # Increase the number of samples
            n_samples = ntest_samples*n_iter

            # Generate samples
            if batch_size_gen == "auto":
                batch_size_gen = 3200000 // ndims
            X_data_test = generate_and_clean_data(targ_dist, n_samples, batch_size_gen, seed_test, gpu = gpu)
            X_data_nf = generate_and_clean_data(nf_dist, n_samples, batch_size_gen, seed_test, gpu = gpu)
            end=timer()
            print("Samples generated in",end-start,"s.")
    
        print("Computing metrics...")
        start=timer()
        ks_means, ks_stds, ks_lists, ad_means, ad_stds, ad_lists, fn_list, wd_means, wd_stds, wd_lists, swd_means, swd_stds, swd_lists = Metrics.ComputeMetricsReduced(X_data_test, X_data_nf, n_iter=n_iter, batch_size=ntest_samples, n_slices=ndims*n_slices_factor, seed=seed_test, verbose=verbose)
        end=timer()
        prediction_time = end-start
        total_time = training_time + prediction_time
        print("Metrics for run",run,"computed in",prediction_time,"s.")

        start=timer()
        hyperparams_dict = {'run_n': [run_n],
                            'seed_train': [seed_train],
                            'seed_test': [seed_test], 
                            'ndims': [ndims],
                            'nsamples': [nsamples],
                            'correlation': [correlation],
                            'nbijectors': [nbijectors],
                            'bijector': [bijector],
                            'spline_knots': [spline_knots],
                            'range_min': [range_min],
                            'hidden_layers': [hidden_layers],
                            'batch_size': [batch_size],
                            'activation': [activation],
                            'eps_regulariser': [eps_regulariser],
                            'regulariser': [regulariser],
                            'dist_seed': [dist_seed],
                            'test_seed': [test_seed],
                            'training_device': [training_device]} 
        results_dict = {'run_n': [run_n],
                        'seed_train': [seed_train],
                        'seed_test': [seed_test], 
                        'ndims': [ndims],
                        'nsamples': [nsamples],
                        'correlation': [correlation],
                        'nbijectors': [nbijectors],
                        'bijector': [bijector],
                        'activation': [activation],
                        'spline_knots': [spline_knots],
                        'range_min': [range_min],
                        'eps_regulariser': [eps_regulariser],
                        'regulariser': [regulariser],
                        'ks_mean': [ks_means],
                        'ks_std': [ks_stds],
                        'ks_list': [ks_lists],
                        'ad_mean': [ad_means],
                        'ad_std': [ad_stds],
                        'ad_list': [ad_lists],
                        'wd_mean': [wd_means],
                        'wd_std': [wd_stds],
                        'wd_list': [wd_lists],
                        'swd_mean': [swd_means],
                        'swd_std': [swd_stds],
                        'swd_list': [swd_lists],
                        'fn_mean': [fn_list],
                        'fn_std': [fn_list],
                        'fn_list': [fn_list],
                        'hidden_layers': [hidden_layers],
                        'batch_size': [batch_size],
                        'epochs_input': [epochs_input],
                        'epochs_output': [epochs_output],
                        'training_time': [training_time],
                        'prediction_time': [prediction_time],
                        'total_time': [total_time],
                        'training_device': [training_device]}
        Utils.results_current(path_to_results,results_dict)
        Utils.save_details_json(hyperparams_dict,results_dict,train_loss_history,val_loss_history,lr_histoty,path_to_results)
        tmp_new = import_csv_with_lists(run)
        tmp_new['ks_list'] = None
        tmp_new['ad_list'] = None
        tmp_new['wd_list'] = None
        tmp_new['swd_list'] = None
        tmp_new['fn_list'] = None
        tmp_new.at[0, 'ks_list'] = ks_lists
        tmp_new.at[0, 'ad_list'] = ad_lists
        tmp_new.at[0, 'fn_list'] = fn_list
        tmp_new.at[0, 'wd_list'] = wd_lists
        tmp_new.at[0, 'swd_list'] = swd_lists
        end=timer()
        print("Results for run",run,"saved in",end-start,"s.")
        end_all=timer()
        print("Run",run,"processed in",end_all-start_all,"s.")
        os.remove(results_bkp_file)
        os.remove(details_bkp_file)
        return tmp_new
    except Exception as e:
        print(e)
        print("Error in run",run,". Restoring backup files.")
        try:
            os.remove(results_source_file)
        except:
            pass
        try:
            os.remove(details_source_file)
        except:
            pass
        os.rename(results_bkp_file, results_source_file)
        os.rename(details_bkp_file, details_source_file)

start_all=timer()
for i in list(range(253,361)):
    run = "/mnt/project_mnt/teo_fs/rtorre/cernbox/git/GitHub/NormalizingFlows/NF4HEP/NormalizingFlowsHD/CMoG/results/MAFN_final/run_"+str(i)+"/results.txt"
    recompute_metrics(run, n_iter = 10, ntest_samples = 100000, batch_size_gen = "auto", n_slices_factor = 2, gpu = True, verbose = True)
end_all=timer()
print("All runs processed in",end_all-start_all,"s.")