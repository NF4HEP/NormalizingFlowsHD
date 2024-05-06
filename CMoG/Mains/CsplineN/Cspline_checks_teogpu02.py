##############################################################################################
######################################### Initialize #########################################
##############################################################################################

visible_devices = [1]
import datetime
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing os...")
import os
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing sys...")
import sys
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing and initializing argparse...")
if not any("ipykernel" in arg for arg in sys.argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visible_devices", help="Set visible devices", nargs='*', type=list, default=visible_devices)
    args = parser.parse_args()
    visible_devices = args.visible_devices if args.visible_devices else visible_devices
    if len(visible_devices) == 0:
        visible_devices = int(visible_devices)
    elif len(visible_devices) == 1:
        if len(visible_devices[0]) == 0:
            visible_devices = int(visible_devices[0])
        else:
            visible_devices = [int(i) for i in visible_devices[0]]
    else:
        visible_devices = [int(i) for i in visible_devices]
print("Visible devices:", visible_devices)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing timer from timeit...")
from timeit import default_timer as timer
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Setting env variables for tf import (only device", visible_devices, "will be available)...")
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in visible_devices])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing numpy...")
import numpy as np
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing pandas...")
import pandas as pd
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing shutil...")
import shutil
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing subprocess...")
import subprocess
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing tensorflow...")
import tensorflow as tf
print("Tensorflow version:", tf.__version__)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing tensorflow_probability...")
import tensorflow_probability as tfp
tfd = tfp.distributions
print("Tensorflow probability version:", tfp.__version__)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing textwrap...")
import textwrap
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing timeit...")
from timeit import default_timer as timer
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing traceback...")
import traceback
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing typing...")
from typing import List, Tuple, Dict, Union, Optional, Any
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Setting tf configs...")
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu_device in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu_device, True)

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "Importing custom module...")

sys.path.append('../../../code')
import Bijectors, Distributions, MixtureDistributions, Plotters, Trainer, Utils # type: ignore
import GenerativeModelsMetrics as GMetrics # type: ignore

def get_gpu_info() -> Optional[List[str]]:
    try:
        gpu_info: str = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"]).decode('utf-8')
        return gpu_info.strip().split('\n')
    except Exception as e:
        print(e)
        return None
gpu_models: Optional[List[str]] = get_gpu_info()
if gpu_models:
    training_device: str = gpu_models[0]
    print("Successfully loaded GPU model: {}".format(training_device))
else:
    training_device = 'undetermined'
    print("Failed to load GPU model. Defaulting to 'undetermined'.")
    
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+":", "All modues imported successfully.")

##############################################################################################
####################################### Helper functions #####################################
##############################################################################################

#tf.config.run_functions_eagerly(True)

def MixtureGaussian(ncomp: int,
                    ndims: int,
                    seed: int = 0) -> tfp.distributions.Mixture:
    targ_dist: tfp.distributions.Mixture = MixtureDistributions.MixMultiNormal1(ncomp,ndims,seed=seed)
    return targ_dist

def get_io_kwargs(path_to_results: str) -> Dict[str,Any]:
    return {'results_path': path_to_results,
            'load_weights': True,
            'load_results': True}
    
def get_data_kwargs(seed: int = 0) -> Dict[str,Any]:
    return {'seed': seed}

def get_compiler_kwargs(lr: float,
                        ignore_nans: bool,
                        nan_threshold: float
                       ) -> Dict[str,Any]:
    compiler_kwargs = {'optimizer': {'class_name': 'Custom>Adam', # this gives the new Adam optimizer
    #compiler_kwargs = {'optimizer': {'class_name': 'Adam', # this gives the new Adam optimizer
                                     'config': {'learning_rate': lr,
                                                'beta_1': 0.9,
                                                'beta_2': 0.999,
                                                'epsilon': 1e-07,
                                                #'clipnorm': 0.5,
                                                'amsgrad': True}},
                       'metrics': [{'class_name': 'MinusLogProbMetric',
                                    'config': {'ignore_nans': ignore_nans, 
                                               'debug_print_mode': False}}],
                       'loss': {'class_name': 'MinusLogProbLoss', 
                                'config': {'name': "MLP", 
                                           'ignore_nans': ignore_nans, 
                                           'nan_threshold': nan_threshold, 
                                           'debug_print_mode': False}}}
    return compiler_kwargs
    
def get_callbacks_kwargs(checkpoint_path: str,
                         es_min_delta: float,
                         es_patience: int,
                         lr_reduce_factor: float,
                         lr_min_delta: float,
                         lr_patience: int,
                         min_lr: float
                         ) -> List[Dict[str,Any]]:
    callbacks_kwargs = [{'class_name': 'PrintEpochInfo',
                         'config': {}},
                        #{'class_name': 'HandleNaNCallback',
                        # 'config': {'checkpoint_path': checkpoint_path,
                        #            'lr_reduction_factor': lr_reduce_factor_on_nan,
                        #            'random_seed_var': np.random.randint(1000000)}},
                        #{'class_name': 'TerminateOnNaNFractionCallback',
                        # 'config': {'threshold': 0.1,
                        #            'validation_data': X_data_val}},
                        {'class_name': 'ModelCheckpoint',
                         'config': {'filepath': checkpoint_path,
                                    'monitor': 'val_loss',
                                    'save_best_only': True,
                                    'save_weights_only': True,
                                    'verbose': 1,
                                    'mode': 'auto',
                                    'save_freq': 'epoch'}},
                        {'class_name': 'EarlyStopping',
                         'config': {'monitor': 'val_loss', 
                                    'min_delta': es_min_delta, 
                                    'patience': es_patience, 
                                    'verbose': 1,
                                    'mode': 'auto', 
                                    'baseline': None, 
                                    'restore_best_weights': True}},
                        {'class_name': 'ReduceLROnPlateau', 
                         'config': {'monitor': 'val_loss', 
                                    'factor': lr_reduce_factor, 
                                    'min_delta': lr_min_delta, 
                                    'patience': lr_patience, 
                                    'min_lr': min_lr}},
                        {'class_name': 'TerminateOnNaN', 'config': {}}
                        ]
    return callbacks_kwargs

def get_fit_kwargs(batch_size: int,
                   epochs_input: int,
                   validation_data: Tuple[Union[np.ndarray,tf.Tensor],Union[np.ndarray,tf.Tensor]],
                   shuffle: bool,
                   verbose: int
                  ) -> Dict[str,Any]:
    fit_kwargs = {'batch_size': batch_size, 
                  'epochs': epochs_input, 
                  'validation_data': validation_data,
                  'shuffle': shuffle, 
                  'verbose': verbose}
    return fit_kwargs

def train_function(seeds: List[int],
                   nsamples: List[int],
                   run_number: int,
                   base_dist: tfp.distributions.Distribution,
                   targ_dist: tfp.distributions.Distribution,
                   hyperparams_dict: Dict[str, Any],
                   n_runs: int,
                   ndims: int,
                   bijector_name: str,
                   nbijectors: int,
                   spline_knots: Union[int,str],
                   range_min: int,
                   hidden_layers: List[int],
                   batch_size: int,
                   epochs_input: int,
                   lr_orig: float,
                   es_min_delta: float,
                   es_patience: int,
                   lr_reduce_factor: float,
                   lr_min_delta: float,
                   lr_patience: int,
                   min_lr: float,
                   activation: str,
                   regulariser: Optional[str],
                   eps_regulariser: float,
                   use_batch_norm: bool,
                   training_device: str,
                   path_to_results: str,
                   checkpoint_path: str,
                   max_retry: int,
                   debug_print_mode: bool,
                   nan_threshold: float,
                  ) -> Tuple[Dict[str, Any], Trainer.Trainer, int, float]:
    succeeded = False
    retry: int = 0
    lr: float = lr_orig
    while not succeeded:
        seed_train: int
        seed_test: int
        seed_dist: int
        seed_metrics: int
        seed_train, seed_test, seed_dist, seed_metrics = seeds
        seed_test = seed_train + 1                     
        Utils.reset_random_seeds(seed = seed_train)
        nsamples_train: int
        nsamples_val: int
        nsamples_test: int
        nsamples_train, nsamples_val, nsamples_test = nsamples
        X_data_train: tf.Tensor
        X_data_val: tf.Tensor
        Y_data_train: tf.Tensor
        Y_data_val: tf.Tensor
        X_data_train, X_data_val, Y_data_train, Y_data_val = Utils.generate_train_data(run_number = run_number,
                                                                                       targ_dist = targ_dist,
                                                                                       nsamples_train = nsamples_train,
                                                                                       nsamples_val = nsamples_val,
                                                                                       seed_train = seed_train)
        #X_data_train = Utils.standardize_data(X_data_train)
        #X_data_val = Utils.standardize_data(X_data_val)
        bijector: tfp.bijectors.Bijector = Bijectors.ChooseBijector(bijector_name = bijector_name,
                                                                    ndims = ndims,
                                                                    spline_knots = spline_knots,
                                                                    nbijectors = nbijectors,
                                                                    range_min = range_min,
                                                                    hidden_layers = hidden_layers,
                                                                    activation = activation,
                                                                    regulariser = regulariser,
                                                                    eps_regulariser = eps_regulariser,
                                                                    use_batch_norm = use_batch_norm)
        Utils.save_bijector_info(path_to_results, bijector)
        print("Building Trainer NFObject.\n")
        NFObject: Trainer.Trainer = Trainer.Trainer(base_distribution = base_dist,
                                                    flow = bijector, 
                                                    x_data_train = X_data_train,
                                                    y_data_train = Y_data_train,
                                                    io_kwargs = get_io_kwargs(path_to_results = path_to_results),
                                                    data_kwargs = get_data_kwargs(seed = seed_train),
                                                    compiler_kwargs = get_compiler_kwargs(lr = lr,
                                                                                          ignore_nans = True,
                                                                                          nan_threshold = nan_threshold),
                                                    callbacks_kwargs = get_callbacks_kwargs(checkpoint_path = checkpoint_path,
                                                                                            es_min_delta = es_min_delta,
                                                                                            es_patience = es_patience,
                                                                                            lr_reduce_factor = lr_reduce_factor,
                                                                                            lr_min_delta = lr_min_delta,
                                                                                            lr_patience = lr_patience,
                                                                                            min_lr = min_lr),
                                                    fit_kwargs = get_fit_kwargs(batch_size = batch_size,
                                                                                epochs_input = epochs_input,
                                                                                validation_data = (X_data_val, Y_data_val),
                                                                                shuffle = True,
                                                                                verbose = 2),
                                                    debug_print_mode = debug_print_mode)
        trainable_params: int = NFObject.trainable_params
        non_trainable_params: int = NFObject.non_trainable_params
        hyperparams_dict = Utils.update_hyperparams_dict(hyperparams_dict = hyperparams_dict,
                                                         run_number = run_number,
                                                         n_runs = n_runs,
                                                         seeds = [seed_train, seed_test, seed_dist, seed_metrics],
                                                         nsamples = [nsamples_train, nsamples_val, nsamples_test],
                                                         ndims = ndims,
                                                         corr = None,
                                                         bijector_name = bijector_name,
                                                         nbijectors = nbijectors,
                                                         spline_knots = spline_knots,
                                                         range_min = range_min,
                                                         hllabel = '-'.join(str(e) for e in hidden_layers),
                                                         trainable_parameters = trainable_params,
                                                         non_trainable_parameters = non_trainable_params,
                                                         batch_size = batch_size,
                                                         epochs_input = epochs_input,
                                                         activation = activation,
                                                         regulariser = regulariser,
                                                         eps_regulariser = eps_regulariser,
                                                         training_device = training_device)
        Utils.save_hyperparams_dict(path_to_results, hyperparams_dict)
        print(f"Training model with initial learning rate {lr}...")
        print("Train first sample:", X_data_train[0]) # type: ignore
        NFObject.train()
        training_time: float = NFObject.training_time # type: ignore
        loss_history = list(NFObject.history['loss'])
        if np.isnan(loss_history).any():
            print("The loss history contains NaN values.")

        if np.isinf(loss_history).any():
            print("The loss history contains Inf values.")

        if len(loss_history) > 0:
            if np.isnan(loss_history).any() or np.isinf(loss_history).any():
                succeeded = False
                seed_train = np.random.randint(1000000)
                lr = lr * lr_reduce_factor_on_nan
                retry = retry + 1
                print(f"Training failed: trying again with seed {seed_train} and lr {lr}.")
            else:
                succeeded = True
                print(f"Training succeeded with seed {seed_train}.")
        else:
            succeeded = False
            seed_train = np.random.randint(1000000)
            lr = lr * lr_reduce_factor_on_nan
            retry = retry + 1
            print(f"Training failed: trying again with seed {seed_train} and lr {lr}.")
            
        if retry > max_retry:
            raise Exception("Training failed for the maximum number of retry.")
        
    return hyperparams_dict, NFObject, seed_train, training_time # type: ignore
    

def prediction_function(hyperparams_dict: Dict[str, Any],
                        results_dict: Dict[str, Any],
                        gpu_models: Optional[List[str]],
                        NFObject: Trainer.Trainer,
                        ndims: int,
                        targ_dist: tfp.distributions.Distribution,
                        seed_test: int,
                        seed_metrics: int,
                        n_iter: int,
                        nsamples_test: int,
                        n_slices_factor: int,
                        dtype: type,
                        max_vectorize: int,
                        mirror_strategy: bool,
                        make_plots: bool,
                        path_to_results: str
                       ) -> Tuple[Dict[str, Any], GMetrics.TwoSampleTestInputs, int, float]:
    start_pred: float = timer()
    t_losses_all: list = list(NFObject.history['loss']) # type: ignore
    v_losses_all: list = list(NFObject.history['val_loss']) # type: ignore
    lr_all: list = list(NFObject.history['lr']) # type: ignore
    epochs_output: int = len(t_losses_all)
    training_time: float = NFObject.training_time # type: ignore
    try:
        print("===========\nComputing predictions\n===========\n")
        print("Computing metrics...")
        start = timer()
        DataInputs: GMetrics.TwoSampleTestInputs = GMetrics.TwoSampleTestInputs(dist_1_input = targ_dist,
                                                                                dist_2_input = NFObject.nf_dist,
                                                                                niter = n_iter,
                                                                                batch_size_test = nsamples_test,
                                                                                batch_size_gen = 100,
                                                                                small_sample_threshold = 1e7,
                                                                                dtype_input = dtype,
                                                                                seed_input = seed_metrics,
                                                                                use_tf = True,
                                                                                mirror_strategy = mirror_strategy,
                                                                                verbose = True)
        #LRMetric: GMetrics.LRMetric = GMetrics.LRMetric(data_input = DataInputs,
        #                                                verbose = True)
        KSTest: GMetrics.KSTest = GMetrics.KSTest(data_input = DataInputs,
                                                  verbose = True)
        SWDMetric: GMetrics.SWDMetric = GMetrics.SWDMetric(data_input = DataInputs,
                                                           nslices = n_slices_factor*ndims,
                                                           seed_slicing = 0,
                                                           verbose = True)
        FNMetric: GMetrics.FNMetric = GMetrics.FNMetric(data_input = DataInputs,
                                                        verbose = True)
        #LRMetric.compute()
        KSTest.compute(max_vectorize = max_vectorize)
        SWDMetric.compute(max_vectorize = max_vectorize)
        FNMetric.compute(max_vectorize = max_vectorize)
        #lr_result: Dict[str, np.ndarray] = LRMetric.Results[-1].result_value
        logprob_ref_ref_sum_list = None#lr_result["logprob_ref_ref_sum_list"].tolist()
        logprob_ref_alt_sum_list = None#lr_result["logprob_ref_alt_sum_list"].tolist()
        logprob_alt_alt_sum_list = None#lr_result["logprob_alt_alt_sum_list"].tolist()
        lik_ratio_list = None#lr_result["lik_ratio_list"].tolist()
        lik_ratio_norm_list = None#lr_result["lik_ratio_norm_list"].tolist()
        ks_result: Dict[str, np.ndarray] = KSTest.Results[-1].result_value
        ks_lists: List[List[float]] = ks_result["statistic_lists"].tolist()
        ks_means: List[float] = ks_result["statistic_means"].tolist()
        ks_stds: List[float] = ks_result["statistic_stds"].tolist()
        swd_result: Dict[str, np.ndarray] = SWDMetric.Results[-1].result_value
        swd_lists: List[List[float]] = swd_result["metric_lists"].tolist()
        swd_means: List[float] = swd_result["metric_means"].tolist()
        swd_stds: List[float] = swd_result["metric_stds"].tolist()
        fn_result: Dict[str, np.ndarray] = FNMetric.Results[-1].result_value
        fn_list: List[float] = fn_result["metric_list"].tolist()
        ad_lists: Optional[List[List[float]]] = None
        ad_means: Optional[List[float]] = None
        ad_stds: Optional[List[float]] = None
        wd_lists: Optional[List[List[float]]] = None
        wd_means: Optional[List[float]] = None
        wd_stds: Optional[List[float]] = None
        end = timer()
        metrics_time = end - start
        print(f"Metrics computed in {metrics_time:.2f} s.")
    except:
        raise Exception("Failed computing metrics")
        try:
            print("===========\nFailed on GPU, re-trying on CPU\n===========\n")
            with tf.device('/device:CPU:0'): # type: ignore
                print("Computing metrics...")
                start = timer()
                DataInputs = GMetrics.TwoSampleTestInputs(dist_1_input = targ_dist,
                                                          dist_2_input = NFObject.nf_dist,
                                                          niter = n_iter,
                                                          batch_size = nsamples_test,
                                                          dtype_input = dtype,
                                                          seed_input = seed_metrics,
                                                          use_tf = True,
                                                          verbose = True)
                LRMetric = GMetrics.LRMetric(data_input = DataInputs,
                                             verbose = True)
                KSTest = GMetrics.KSTest(data_input = DataInputs,
                                         verbose = True)
                SWDMetric = GMetrics.SWDMetric(data_input = DataInputs,
                                               verbose = True)
                FNMetric = GMetrics.FNMetric(data_input = DataInputs,
                                             verbose = True)
                LRMetric.compute()
                KSTest.compute(max_vectorize = max_vectorize)
                SWDMetric.compute(nslices = n_slices_factor*ndims)
                FNMetric.compute(max_vectorize = max_vectorize)
                lr_result = LRMetric.Results[-1].result_value
                logprob_ref_ref_sum_list = lr_result["logprob_ref_ref_sum_list"].tolist()
                logprob_ref_alt_sum_list = lr_result["logprob_ref_alt_sum_list"].tolist()
                logprob_alt_alt_sum_list = lr_result["logprob_alt_alt_sum_list"].tolist()
                lik_ratio_list = lr_result["lik_ratio_list"].tolist()
                lik_ratio_norm_list = lr_result["lik_ratio_norm_list"].tolist()
                ks_result = KSTest.Results[-1].result_value
                ks_lists = ks_result["statistic_lists"].tolist()
                ks_means = ks_result["statistic_means"].tolist()
                ks_stds = ks_result["statistic_stds"].tolist()
                swd_result = SWDMetric.Results[-1].result_value
                swd_lists = swd_result["metric_lists"].tolist()
                swd_means = swd_result["metric_means"].tolist()
                swd_stds = swd_result["metric_stds"].tolist()
                fn_result = FNMetric.Results[-1].result_value
                fn_list = fn_result["metric_list"].tolist()
                ad_lists = None
                ad_means = None
                ad_stds = None
                wd_lists = None
                wd_means = None
                wd_stds = None
                end = timer()
                metrics_time = end - start
                print(f"Metrics computed in {metrics_time:.2f} s.")
        except:
            print("===========\nFailed computing metrics\n===========\n")
            logprob_ref_ref_sum_list = []
            logprob_ref_alt_sum_list = []
            logprob_alt_alt_sum_list = []
            lik_ratio_list = []
            lik_ratio_norm_list = []
            ks_means = []
            ks_stds = []
            ks_lists = []
            ad_means = []
            ad_stds = []
            ad_lists = []
            fn_list = []
            wd_means = []
            wd_stds = []
            wd_lists = []
            swd_means = []
            swd_stds = []
            swd_lists = []
            metrics_time = 0.
    if make_plots:
        try:
            start = timer()
            Plotters.train_plotter(t_losses_all,v_losses_all,path_to_results)
            X_data_test: tf.Tensor = DataInputs.dist_1_num[:nsamples_test] # type: ignore
            X_data_nf: tf.Tensor = DataInputs.dist_2_num[:nsamples_test] # type: ignore
            Plotters.cornerplotter(X_data_test.numpy(),X_data_nf.numpy(),path_to_results,ndims,norm=True) # type: ignore
            Plotters.marginal_plot(X_data_test.numpy(),X_data_nf.numpy(),path_to_results,ndims) # type: ignore
            #Plotters.sample_plotter(X_data_test,nf_dist,path_to_results)
            end = timer()
            plots_time: float = end - start
            print(f"Plots done in {plots_time:.2f} s.")
        except Exception as ex:
            # Get current system exception
            ex_type, ex_value, ex_traceback = sys.exc_info()
            # Extract unformatter stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback) # type: ignore
            # Format stacktrace
            stack_trace = list()
            for trace in trace_back:
                stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
            ex_type_str = f"Exception type: {ex_type.__name__}" # type: ignore
            print(textwrap.dedent(f"""\
                ===========
                print("===========\nFailed to plot\n===========\n")
                {ex_type_str}
                Exception message: {ex_value}
                Stack trace: {stack_trace}
                ===========
                """))
        
    end_pred: float = timer()
    prediction_time: float = end_pred - start_pred
    total_time: float = training_time + prediction_time
    results_dict = Utils.update_results_dict(results_dict = results_dict,
                                             hyperparams_dict = hyperparams_dict,
                                             train_loss_history = t_losses_all,
                                             val_loss_history = v_losses_all,
                                             lr_history = lr_all,
                                             epochs_output = epochs_output,
                                             training_time = training_time,
                                             prediction_time = prediction_time,
                                             total_time = total_time,
                                             logprob_ref_ref_sum_list = logprob_ref_ref_sum_list,
                                             logprob_ref_alt_sum_list = logprob_ref_alt_sum_list,
                                             logprob_alt_alt_sum_list = logprob_alt_alt_sum_list,
                                             lik_ratio_list = lik_ratio_list,
                                             lik_ratio_norm_list = lik_ratio_norm_list,
                                             ks_means = ks_means,
                                             ks_stds = ks_stds,
                                             ks_lists = ks_lists,
                                             ad_means = ad_means,
                                             ad_stds = ad_stds,
                                             ad_lists = ad_lists,
                                             fn_list = fn_list,
                                             wd_means = wd_means,
                                             wd_stds = wd_stds,
                                             wd_lists = wd_lists,
                                             swd_means = swd_means,
                                             swd_stds = swd_stds,
                                             swd_lists = swd_lists
                                             )
    return results_dict, DataInputs, prediction_time, total_time # type: ignore

##############################################################################################
################################## Parameters initialization #################################
##############################################################################################

### Initialize number of components ###
ncomp: int = 3

### Initialize hyperparameters lists ###
ndims: int = 16
nbijectors: int = 2
hidden_layers: List[int] = [128, 128, 128]
seed_train: int = 869

### Initialize nsamples inputs ###
nsamples_train: int = 100000
nsamples_val: int = 30000
nsamples_test: int = 100000

### Initialize seeds inputs ###
seed_test: int = 0 # overwritten in the loop by seed_train + 1
seed_dist: int = 0
seed_metrics: int = seed_test

### Initialize bijector inputs ###
bijector_name: str = 'CsplineN'
range_min: int = -16
spline_knots = 8

### Initialize NN hyperparameters ###
activation: str = 'relu'
regulariser: Optional[str] = None
eps_regulariser: float = 0.
use_batch_norm: bool = False

### Initialzie training hyperparameters ###
epochs_input: int = 1000
batch_size: int = 512
nan_threshold: float = 0.01
max_retry: int = 10
debug_print_mode: bool = True

### Initialize optimizer hyperparameters ###
lr_orig: float = 0.001

### Initialize callbacks hyperparameters ###
es_min_delta: float = .0001
es_patience: int = 100
lr_min_delta: float = .0001
lr_patience: int = 50
lr_reduce_factor: float = .5
lr_reduce_factor_on_nan: float = float(1/3)
min_lr: float = 1e-6

### Initialize parameters for inference ###
n_iter: int = 10
n_slices_factor: int = 2
dtype: type = tf.float32
max_vectorize: int = 10
mirror_strategy = False
make_plots = True

### Initialize old variables for backward compatibility
corr: Optional[str] = None

### Initialize old variables for backward compatibility
corr: Optional[str] = None

### Initialize dictionaries ###
results_dict: Dict[str, Any] = Utils.init_results_dict()
hyperparams_dict: Dict[str, Any] = Utils.init_hyperparams_dict()

### Initialize output dir ###
mother_output_dir: str = Utils.define_dir('../../results/CsplineN_test/')

### Create 'log' file ####
log_file_name: str = Utils.create_log_file(mother_output_dir, results_dict)

##############################################################################################
####################################### Training loop ########################################
##############################################################################################
run_number: int = 0
n_runs = 1
start_global: float = timer()
targ_dist: tfp.distributions.Distribution = MixtureGaussian(ncomp = ncomp, ndims = ndims, seed = seed_dist)
base_dist: tfp.distributions.Distribution = Distributions.gaussians(ndims)
start_run: float = timer()
hllabel: str = '-'.join(str(e) for e in hidden_layers)
run_number = run_number + 1
results_dict_txt_saved: bool = False
results_dict_json_saved: bool = False
results_log_saved: bool = False
path_to_results: str
to_run: bool
path_to_results, to_run = Utils.define_run_dir(mother_output_dir+'run_'+str(run_number)+'/',
                                               force = "delete",
                                               bkp = False)
if to_run:
    #try:
    dummy_file_path: str = os.path.join(path_to_results,'running.txt')
    with open(dummy_file_path, 'w') as f:
        pass
    path_to_weights: str = Utils.define_dir(os.path.join(path_to_results, 'weights'))
    checkpoint_path: str = os.path.join(path_to_weights, 'best_weights.h5')
    ########### Model train ###########
    NFObject: Trainer.Trainer
    #tf.data.experimental.enable_debug_mode()
    hyperparams_dict, NFObject, seed_train, training_time = train_function(seeds = [seed_train, seed_test, seed_dist, seed_metrics],
                                                                           nsamples = [nsamples_train, nsamples_val, nsamples_test],
                                                                           run_number = run_number,
                                                                           base_dist = base_dist,
                                                                           targ_dist = targ_dist,
                                                                           hyperparams_dict = hyperparams_dict,
                                                                           n_runs = n_runs,
                                                                           ndims = ndims,
                                                                           bijector_name = bijector_name,
                                                                           nbijectors = nbijectors,
                                                                           spline_knots = spline_knots,
                                                                           range_min = range_min,
                                                                           hidden_layers = hidden_layers,
                                                                           batch_size = batch_size,
                                                                           epochs_input = epochs_input,
                                                                           lr_orig = lr_orig,
                                                                           es_min_delta = es_min_delta,
                                                                           es_patience = es_patience,
                                                                           lr_reduce_factor = lr_reduce_factor,
                                                                           lr_min_delta = lr_min_delta,
                                                                           lr_patience = lr_patience,
                                                                           min_lr = min_lr,
                                                                           activation = activation,
                                                                           regulariser = regulariser,
                                                                           eps_regulariser = eps_regulariser,
                                                                           use_batch_norm = use_batch_norm,
                                                                           training_device = training_device,
                                                                           path_to_results = path_to_results,
                                                                           checkpoint_path = checkpoint_path,
                                                                           max_retry = max_retry,
                                                                           debug_print_mode = debug_print_mode,
                                                                           nan_threshold = nan_threshold)
     
    print(f"Model trained in {training_time:.2f} s.\n") # type: ignore
    ########## Model prediction ###########
    results_dict, DataInputs, prediction_time, total_time = prediction_function(hyperparams_dict = hyperparams_dict,
                                                                                results_dict = results_dict,
                                                                                gpu_models = gpu_models,
                                                                                NFObject = NFObject, # type: ignore
                                                                                ndims = ndims,
                                                                                targ_dist = targ_dist,
                                                                                seed_test = seed_test,
                                                                                seed_metrics = seed_metrics,
                                                                                n_iter = n_iter,
                                                                                nsamples_test = nsamples_test,
                                                                                n_slices_factor = n_slices_factor,
                                                                                dtype = dtype,
                                                                                max_vectorize = max_vectorize,
                                                                                mirror_strategy = mirror_strategy,
                                                                                make_plots = make_plots,
                                                                                path_to_results = path_to_results)
    ########### Save results ###########
    Utils.save_results_current_run_txt(path_to_results, results_dict)
    results_dict_txt_saved = True
    print("results.txt saved")
    Utils.save_results_current_run_json(path_to_results, results_dict)
    results_dict_json_saved = True
    print("results.json saved")
    Utils.save_results_log(log_file_name, results_dict)
    results_log_saved = True
    print("Results log saved")
    print(f"Model predictions computed in {prediction_time:.2f} s.")
    end_run: float = timer()
    total_time_run=end_run-start_run
    print(textwrap.dedent(f"""\
        ===========
        Run {run_number}/{n_runs} done in {total_time_run:.2f} s.
        ===========
        """))
    run = run + 1
    try:
        os.remove(dummy_file_path)
    except:
        pass
    dummy_file_path = os.path.join(path_to_results,'done.txt')
    with open(dummy_file_path, 'w') as f:
        pass
    #except Exception as ex:
    #    try:
    #        os.remove(dummy_file_path)
    #    except:
    #        pass
    #    # Get current system exception
    #    ex_type, ex_value, ex_traceback = sys.exc_info()
    #    # Extract unformatter stack traces as tuples
    #    trace_back = traceback.extract_tb(ex_traceback) # type: ignore
    #    # Format stacktrace
    #    stack_trace = list()
    #    for trace in trace_back:
    #        stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
    #    if not results_dict_txt_saved:
    #        results_dict = Utils.update_results_dict(results_dict = results_dict,
    #                                                 hyperparams_dict = hyperparams_dict)
    #        Utils.save_results_current_run_txt(path_to_results, results_dict)
    #    if not results_dict_json_saved:
    #        Utils.save_results_current_run_json(path_to_results, results_dict)
    #    if not results_log_saved:
    #        Utils.save_results_log(log_file_name, results_dict)
    #    ex_type_str = f"Exception type: {ex_type.__name__}" # type: ignore
    #    print(textwrap.dedent(f"""\
    #        ===========
    #        Run {run_number}/{n_runs} failed.
    #        {ex_type_str}
    #        Exception message: {ex_value}
    #        Stack trace: {stack_trace}
    #        ===========
    #        """))
else:
    print(textwrap.dedent(f"""\
        ===========
        Run {run_number}/{n_runs} already exists. Skipping it.
        ===========
        """))
keys_to_remove = ['ks_lists', 'ad_lists', 'fn_list', 'wd_lists', 'swd_lists', 'train_loss_history', 'val_loss_history', 'lr_history']
dict_copy: Dict[str, Any] = {k: v for k, v in results_dict.items() if k not in keys_to_remove}
results_frame: pd.DataFrame = pd.DataFrame(dict_copy)
results_last_run_file: str = os.path.join(mother_output_dir,'results_last_run.txt')
results_frame.to_csv(results_last_run_file,index=False)
end_global: float = timer()
print(f"Everything done in {end_global-start_global:.2f} s.\n")