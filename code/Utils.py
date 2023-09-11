import os
import numpy as np
from timeit import default_timer as timer
import datetime
import codecs
import textwrap
import random
import json
import shutil
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb= tfp.bijectors
from tensorflow.keras.layers import Input # type: ignore
from tensorflow.keras import Model # type: ignore
import pandas as pd
from typing import List, Tuple, Dict, Callable, Union, Optional, Any, Type

import MixtureDistributions

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def init_hyperparams_dict():
    hyperparams_dict: Dict[str,Any] = {'run_n': [],
                                       'seed_train': [],
                                       'seed_test': [],
                                       'seed_dist': [],
                                       'seed_metrics': [],
                                       'ndims':[],
                                       'nsamples_train': [],
                                       'nsamples_val': [],
                                       'nsamples_test': [],
                                       'correlation':[],
                                       'nbijectors':[],
                                       'bijector':[],
                                       'spline_knots':[],
                                       'range_min':[],
                                       'hidden_layers':[],
                                       'trainable_parameters': [],
                                       'non_trainable_parameters': [],
                                       'batch_size':[],
                                       'activation':[],
                                       'regulariser':[],
                                       'eps_regulariser':[],
                                       'training_device':[],
                                       'epochs_input': []} 
    return hyperparams_dict

def init_results_dict():
    hyperparams_dict: Dict[str,Any] = init_hyperparams_dict()
    results_dict: Dict[str,Any] = {'epochs_output': [],
                                   'training_time': [],
                                   'prediction_time': [],
                                   'total_time': [],
                                   'best_train_loss': [],
                                   'best_val_loss': [],	
                                   'best_train_epoch': [],
                                   'best_val_epoch': [],
                                   'train_loss_history': [],
                                   'val_loss_history': [],
                                   'lr_history': [],
                                   'logprob_ref_ref_sum_list': [],
                                   'logprob_ref_alt_sum_list': [],
                                   'logprob_alt_alt_sum_list': [],
                                   'lik_ratio_list': [],
                                   'lik_ratio_norm_list': [],
                                   'ks_means': [],
                                   'ks_stds': [],
                                   'ks_lists': [],
                                   'ad_means': [],
                                   'ad_stds': [],
                                   'ad_lists': [],
                                   'fn_list': [],
                                   'wd_means': [],
                                   'wd_stds': [],
                                   'wd_lists': [],
                                   'swd_means': [],
                                   'swd_stds': [],
                                   'swd_lists': []}
    return {**hyperparams_dict, **results_dict}

def print_run_info(run_number: int,
                   n_runs: int,
                   ndims: int,
                   seed_train: int,
                   nsamples: List[int],
                   activation: str,
                   bijector_name: str,
                   nbijectors: int,
                   spline_knots: int,
                   range_min: float,
                   batch_size: int,
                   hidden_layers: str,
                   trainable_parameters: int,
                   epochs_input: int,
                   training_device: str
                  ) -> None:
    nsamples_train: int
    nsamples_val: int
    nsamples_test: int
    nsamples_train, nsamples_val, nsamples_test = nsamples
    print(textwrap.dedent(f"""\
        ===============
        Running {run_number}/{n_runs} with hyperparameters:
        timestamp = {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}
        ndims = {ndims}
        seed_train = {seed_train}
        nsamples_train = {nsamples_train}
        nsamples_val = {nsamples_val}
        nsamples_test = {nsamples_test}
        bijector = {bijector_name}
        nbijectors = {nbijectors}
        spline_knots = {spline_knots}
        range_min = {range_min}
        hidden_layers = {hidden_layers}
        trainable_parameters = {trainable_parameters}
        epochs_input = {epochs_input}
        batch_size = {batch_size}
        activation = {activation}
        training_device = {training_device}
        ===============
        """))
    
def update_hyperparams_dict(hyperparams_dict: Dict[str, Any],
                            run_number: int,
                            n_runs: int,
                            seeds: List[int],
                            nsamples: List[int],
                            ndims: int,
                            corr: Optional[str],
                            bijector_name: str,
                            nbijectors: int,
                            spline_knots: int,
                            range_min: float,
                            hllabel: str,
                            trainable_parameters: int,
                            non_trainable_parameters: int,
                            batch_size: int,
                            epochs_input: int,
                            activation: str,
                            regulariser: Optional[str],
                            eps_regulariser: float,
                            training_device: str
                           ) -> Dict[str, Any]:
    """
    Function that writes hyperparameters values to a dictionary and saves it to the hyperparam.txt file.
    """
    seed_train, seed_test, seed_dist, seed_metrics = seeds
    nsamples_train, nsamples_val, nsamples_test = nsamples
    # Define keys and values in a list of tuples
    keys_and_values = [
        ('run_n', run_number),
        ('seed_train', seed_train),
        ('seed_test', seed_test),
        ('seed_dist', seed_dist),
        ('seed_metrics', seed_metrics),
        ('ndims', ndims),
        ('nsamples_train', nsamples_train),
        ('nsamples_val', nsamples_val),
        ('nsamples_test', nsamples_test),
        ('correlation', corr),
        ('bijector', bijector_name),
        ('nbijectors', nbijectors),
        ('spline_knots', spline_knots),
        ('range_min', range_min),
        ('hidden_layers', hllabel),
        ('trainable_parameters', trainable_parameters),
        ('non_trainable_parameters', non_trainable_parameters),
        ('batch_size', batch_size),
        ('epochs_input', epochs_input),
        ('activation', activation),
        ('regulariser', regulariser),
        ('eps_regulariser', eps_regulariser),
        ('training_device', training_device)]
    # Append to hyperparams_dict
    for key, value in keys_and_values:
        hyperparams_dict.setdefault(key, []).append(value)
    print_run_info(run_number, 
                   n_runs, 
                   ndims, 
                   seed_train, 
                   nsamples,
                   activation, 
                   bijector_name, 
                   nbijectors, 
                   spline_knots, 
                   range_min, 
                   batch_size, 
                   hllabel,
                   trainable_parameters,
                   epochs_input,
                   training_device)
    return hyperparams_dict
    
def update_results_dict(results_dict: Dict[str, Any],
                        hyperparams_dict: Dict[str, Any],
                        train_loss_history: Optional[List[float]] = None,
                        val_loss_history: Optional[List[float]] = None,
                        lr_history: Optional[List[float]] = None,
                        epochs_output: Optional[int] = None,
                        training_time: Optional[int] = None,
                        prediction_time: Optional[int] = None,
                        total_time: Optional[int] = None,
                        logprob_ref_ref_sum_list: Optional[List[float]] = None,
                        logprob_ref_alt_sum_list: Optional[List[float]] = None,
                        logprob_alt_alt_sum_list: Optional[List[float]] = None,
                        lik_ratio_list: Optional[List[float]] = None,
                        lik_ratio_norm_list: Optional[List[float]] = None,
                        ks_means: Optional[List[float]] = None,
                        ks_stds: Optional[List[float]] = None,
                        ks_lists: Optional[List[List[float]]] = None,
                        ad_means: Optional[List[float]] = None,
                        ad_stds: Optional[List[float]] = None,
                        ad_lists: Optional[List[List[float]]] = None,
                        fn_list: Optional[List[float]] = None,
                        wd_means: Optional[List[float]] = None,
                        wd_stds: Optional[List[float]] = None,
                        wd_lists: Optional[List[List[float]]] = None,
                        swd_means: Optional[List[float]] = None,
                        swd_stds: Optional[List[float]] = None,
                        swd_lists: Optional[List[List[float]]] = None,
                       ) -> Dict[str, Any]:
    """
    Function that writes results to a dictionary.
    """
    # Copy hyperparams_dict into results_dict
    results_dict.update(hyperparams_dict)
    # Find best train loss and corresponding epoch
    if train_loss_history:
        best_train_loss = np.min(train_loss_history)
        try:
            position_best_train_loss = np.where(train_loss_history == best_train_loss)[0][0]
        except:
            try:
                position_best_train_loss = np.where(train_loss_history == best_train_loss)[0]
            except:
                position_best_train_loss = None
    else:
        best_train_loss = None
        position_best_train_loss = None
    # Find best val loss and corresponding epoch
    if val_loss_history:
        best_val_loss = np.min(val_loss_history)
        try:
            position_best_val_loss = np.where(val_loss_history == best_val_loss)[0][0]
        except:
            try:
                position_best_val_loss = np.where(val_loss_history == best_val_loss)[0]
            except:
                position_best_val_loss = None
    else:
        best_val_loss = None
        position_best_val_loss = None
    # Create a list of keys and values
    keys_and_values = [
        ('train_loss_history', train_loss_history),
        ('val_loss_history', val_loss_history),
        ('lr_history', lr_history),
        ('best_train_loss', best_train_loss),
        ('best_val_loss', best_val_loss),
        ('best_train_epoch', position_best_train_loss),
        ('best_val_epoch', position_best_val_loss),
        ('epochs_output', epochs_output),
        ('training_time', training_time),
        ('prediction_time', prediction_time),
        ('total_time', total_time),
        ('logprob_ref_ref_sum_list', logprob_ref_ref_sum_list),
        ('logprob_ref_alt_sum_list', logprob_ref_alt_sum_list),
        ('logprob_alt_alt_sum_list', logprob_alt_alt_sum_list),
        ('lik_ratio_list', lik_ratio_list),
        ('lik_ratio_norm_list', lik_ratio_norm_list),
        ('ks_means', ks_means),
        ('ks_stds', ks_stds),
        ('ks_lists', ks_lists),
        ('ad_means', ad_means),
        ('ad_stds', ad_stds),
        ('ad_lists', ad_lists),
        ('fn_list', fn_list),
        ('wd_means', wd_means),
        ('wd_stds', wd_stds),
        ('wd_lists', wd_lists),
        ('swd_means', swd_means),
        ('swd_stds', swd_stds),
        ('swd_lists', swd_lists)]
    # Append to results_dict
    for key, value in keys_and_values:
        results_dict.setdefault(key, []).append(value)
    return results_dict

def save_hyperparams_dict(path_to_results: str, hyperparams_dict: Dict[str, Any]) -> None:
    """
    Function that writes hyperparameters values to a dictionary and saves it to the hyperparam.txt file.
    """
    hyperparams_frame = pd.DataFrame(hyperparams_dict)
    hyperparams_txt_file = os.path.join(path_to_results, 'hyperparams.txt')
    hyperparams_frame.to_csv(hyperparams_txt_file, index=False)

def save_results_current_run_txt(path_to_results: str, results_dict: Dict[str, Any]) -> None:
    """
    Function that writes results of the current run to the results.txt file.
    """
    # Remove specific keys from a copy of the results_dict
    keys_to_remove = ['ks_lists', 'ad_lists', 'fn_list', 'wd_lists', 'swd_lists', 'train_loss_history', 'val_loss_history', 'lr_history']
    dict_copy = {k: v for k, v in results_dict.items() if k not in keys_to_remove}
    # Generate header and data strings
    header = ','.join(dict_copy.keys())
    data_string = ','.join(str(value[-1]) for value in dict_copy.values())
    # Write to file
    results_txt_file = os.path.join(path_to_results, 'results.txt')
    with open(results_txt_file, 'w') as f:
        f.write(f"{header}\n{data_string}\n")
        
def save_results_current_run_json(path_to_results: str, results_dict: Dict[str, Any]) -> None:
    """ 
    Function that writes results of the current run to the results.json file.
    """
    dict_to_save = convert_types_dict({k: v[-1] for k, v in results_dict.items()})
    results_json_file = os.path.join(path_to_results, 'results.json')
    with codecs.open(results_json_file, "w", encoding="utf-8") as f:
        json.dump(dict_to_save, f, separators=(",", ":"), indent=4)

def save_results_log(log_file_name: str, 
                     results_dict: Dict[str, Any]
                    ) -> None:
    """
    Logger that writes results of each run to a common log file.
    """
    # Remove specific keys from a copy of the results_dict
    keys_to_remove = ['ks_lists', 'ad_lists', 'fn_list', 'wd_lists', 'swd_lists', 'train_loss_history', 'val_loss_history', 'lr_history']
    dict_copy = {k: v for k, v in results_dict.items() if k not in keys_to_remove}
    # Create a string list of the last values in dict_copy's values
    string_list = [str(value[-1]) for value in dict_copy.values()]
    # Combine into a single string
    string = ','.join(string_list)
    # Write to file
    with open(log_file_name, 'a') as log_file:
        log_file.write(string + '\n')
        
def save_bijector_info(path_to_results: str,
                       bijector: tfp.bijectors.Bijector
                      ) -> None:
    """
    Function that saves the bijector.
    """
    file_path = os.path.join(path_to_results, 'bijector_chain.txt')
    with open(file_path, 'w') as bij_out_file:
        for bij in list(bijector.bijectors):
            bij_out_file.write(f"{bij.name}\n")
            bij_out_file.write(f"{bij.parameters}\n")

def define_dir(dir: str) -> str:
    try:
        os.mkdir(dir)
    except:
        print('Directory',dir,'already exists.')
    return dir

def backup_existing_dir(dir: str, 
                        dir_bkp: str
                       ) -> None:
    if not os.path.exists(dir_bkp):
        shutil.move(dir, dir_bkp)
        print("Old run backed up.")
    else:
        print("Backup already exists.")

def handle_existing_dir(dir: str, 
                        force: str, 
                        bkp: bool) -> bool:
    if force == "skip":
        print("Skipping it.")
        return False
    elif force == "continue":
        print("Continuing it. Ensure that you have properly set the loading parameters load_weights and load_history.")
        return True
    elif force == "delete":
        print("Deleting old run and running again.")
        if bkp:
            dir_bkp = (dir[:-1] if dir.endswith("/") else dir) + '_bkp'
            backup_existing_dir(dir, dir_bkp)
        if os.path.exists(dir):  # Re-check the existence of the directory before removing
            shutil.rmtree(dir)
            print("Old run deleted.")
        os.makedirs(dir, exist_ok=True)  # Explicitly creating the new directory here.
        return True
    else:
        raise ValueError("force must be one of 'delete', 'continue', 'skip'.")

def define_run_dir(dir: str, 
                   force: str = "delete", 
                   bkp: bool = True
                  ) -> Tuple[str, bool]:
    try:
        os.makedirs(dir, exist_ok=False)
        to_run = True
    except FileExistsError:
        print('Directory',dir,'already exists.')
        to_run = handle_existing_dir(dir, force, bkp)
        
    return dir, to_run

def generate_train_data(run_number: int, 
                        targ_dist: tfp.distributions.Distribution,
                        nsamples_train: int, 
                        nsamples_val: int, 
                        seed_train: int
                       ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Generate data and return it.
    """
    print(f"===========\nGenerating train data for run {run_number}.\n===========")
    start = timer()
    X_data_train: tf.Tensor = targ_dist.sample(nsamples_train, seed=seed_train).numpy()
    Y_data_train: tf.Tensor = tf.zeros((X_data_train.shape[0], 0), dtype=X_data_train.dtype)
    X_data_val: tf.Tensor = targ_dist.sample(nsamples_val, seed=seed_train).numpy()
    Y_data_val: tf.Tensor = tf.zeros((X_data_val.shape[0], 0), dtype=X_data_val.dtype)
    end = timer()
    print(f"Train data generated in {end - start:.2f} s.\n")
    return X_data_train, X_data_val, Y_data_train, Y_data_val

#def saver(nf_dist,path_to_weights,iter):
#    """
#    Function that saves the model.
#    """
#    for j in range(len(list(nf_dist.bijector.bijectors))):
#        weights_dir=path_to_weights+'iter_'+str(iter)
#        try:
#            os.mkdir(weights_dir)
#        except:
#            print(weights_dir+' file exists')
#        name=nf_dist.bijector.bijectors[j].name
#        if name=='MAFspline':
#            weights=nf_dist.bijector.bijectors[j].parameters.get('shift_and_log_scale_fn').get_weights()            
#            weights_file = open(weights_dir+'/'+name+'_'+str(j)+'.pkl', "wb")
#            pickle.dump(weights,weights_file)
#        else:
#            continue
#    return



def flatten_list(lst):
    out = []
    for item in lst:
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(flatten_list(item))
        else:
            out.append(item)
    return out

def convert_types_dict(d):
    dd = {}
    for k, v in d.items():
        if isinstance(v, dict):
            dd[k] = convert_types_dict(v)
        elif type(v) == np.ndarray:
            dd[k] = v.tolist()
        elif type(v) == list:
            if str in [type(q) for q in flatten_list(v)]:
                dd[k] = np.array(v, dtype=object).tolist()
            else:
                dd[k] = np.array(v).tolist()
        else:
            dd[k] = np.array(v).tolist()
    return dd

def save_details_json(hyperparams_dict,results_dict,train_loss_history,val_loss_history,lr_history,path_to_results):
    """ Save results and hyperparameters json
    """
    if val_loss_history is None:
        val_loss_history = []
    if train_loss_history is None:
        train_loss_history = []
    if lr_history is None:
        lr_history = []
    train_loss_history = np.array(train_loss_history)
    val_loss_history = np.array(val_loss_history)
    lr_history = np.array(lr_history)
    if val_loss_history.tolist() != []:
        best_val_loss = np.min(val_loss_history)
        try:
            position_best_val_loss = np.where(val_loss_history == best_val_loss)[0][0]
        except:
            try:
                position_best_val_loss = np.where(val_loss_history == best_val_loss)[0]
            except:
                position_best_val_loss = None
        if position_best_val_loss is not None:
            best_train_loss = train_loss_history[position_best_val_loss]
        else:
            best_train_loss = None
    else:
        best_val_loss = None
        position_best_val_loss = None
        best_train_loss = None
    hd={}
    rd={}
    for k in hyperparams_dict.keys():
        hd[k] = hyperparams_dict[k][-1]
    for k in results_dict.keys():
        rd[k] = results_dict[k][-1]
    details_dict = {**hd,**rd,
                    "train_loss_history": train_loss_history.tolist(),
                    "val_loss_history": val_loss_history.tolist(),
                    "lr_history": lr_history.tolist(), 
                    "best_train_loss": best_train_loss,
                    "best_val_loss": best_val_loss,
                    "best_epoch": position_best_val_loss}
    dictionary = convert_types_dict(details_dict)
    with codecs.open(path_to_results+'details.json', "w", encoding="utf-8") as f:
        json.dump(dictionary, f, separators=(",", ":"), indent=4)

def create_log_file(mother_output_dir,results_dict):
    dict_copy = results_dict.copy()
    dict_copy.pop('ks_lists')
    dict_copy.pop('ad_lists')
    dict_copy.pop('fn_list')
    dict_copy.pop('wd_lists')
    dict_copy.pop('swd_lists')
    log_file_name=mother_output_dir+'log_file_eternal.txt'
    if os.path.isfile(log_file_name)==False:
        log_file=open(log_file_name,'w')
        header=','.join(list(dict_copy.keys()))
        log_file.write(header)
        log_file.write('\n')
        log_file.close()
    return log_file_name

def generate_and_clean_data_simple(dist, n_samples, batch_size, dtype, seed):
    if dtype is None:
        dtype = tf.float32
    X_data = []
    total_samples = 0

    while total_samples < n_samples:
        try:
            batch = dist.sample(batch_size, seed=seed)

            # Find finite values
            finite_indices = tf.reduce_all(tf.math.is_finite(batch), axis=1)

            # Warn the user if there are any non-finite values
            n_nonfinite = tf.reduce_sum(tf.cast(~finite_indices, tf.int32))
            if n_nonfinite > 0:
                print(f"Warning: Removed {n_nonfinite} non-finite values from the batch")

            # Select only the finite values
            finite_batch = batch.numpy()[finite_indices.numpy()].astype(dtype.as_numpy_dtype)

            X_data.append(finite_batch)
            total_samples += len(finite_batch)
        except (RuntimeError, tf.errors.ResourceExhaustedError):
            # If a RuntimeError or a ResourceExhaustedError occurs (possibly due to OOM), halve the batch size
            batch_size = batch_size // 2
            print("Warning: Batch size too large. Halving batch size to {}".format(batch_size),"and retrying.")
            if batch_size == 0:
                raise RuntimeError("Batch size is zero. Unable to generate samples.")

    return np.concatenate(X_data, axis=0)[:n_samples]

def generate_and_clean_data_mirror(dist, n_samples, batch_size, dtype, seed):
    if dtype is None:
        dtype = tf.float32
        
    strategy = tf.distribute.MirroredStrategy()  # setup the strategy

    # Create a new random generator with a seed
    rng = tf.random.Generator.from_seed(seed)

    # Compute the global batch size using the number of replicas.
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    @tf.function
    def sample_and_clean(dist, batch_size, rng):
        new_seed = rng.make_seeds(2)[0]
        new_seed = tf.cast(new_seed, tf.int32)  # Convert the seed to int32
        batch = dist.sample(batch_size, seed=new_seed)
        finite_indices = tf.reduce_all(tf.math.is_finite(batch), axis=1)
        finite_batch = tf.boolean_mask(batch, finite_indices)
        return finite_batch, tf.shape(finite_batch)[0]

    total_samples = 0
    samples = []

    with strategy.scope():
        while total_samples < n_samples:
            try:
                per_replica_samples, per_replica_sample_count = strategy.run(sample_and_clean, args=(dist, global_batch_size, rng))
                # concatenate samples on each replica and append to list
                per_replica_samples_concat = tf.concat(strategy.experimental_local_results(per_replica_samples), axis=0)
                samples.append(per_replica_samples_concat)
                total_samples += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_sample_count, axis=None)
                #print("Generated {} samples".format(total_samples))
            except (RuntimeError, tf.errors.ResourceExhaustedError):
                # If a RuntimeError or a ResourceExhaustedError occurs (possibly due to OOM), halve the batch size
                global_batch_size = global_batch_size // 2
                print("Warning: Batch size too large. Halving batch size to {}".format(global_batch_size),"and retrying.")
                if global_batch_size == 0:
                    raise RuntimeError("Batch size is zero. Unable to generate samples.")

    # concatenate all samples
    samples = tf.concat(samples, axis=0)

    # return the first `n_samples` samples
    return samples[:n_samples]

def generate_and_clean_data(dist, n_samples, batch_size, dtype, seed, mirror_strategy = False):
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if mirror_strategy:
        if len(gpu_devices) > 1:
            return generate_and_clean_data_mirror(dist, n_samples, batch_size, dtype, seed)
        else:
            return generate_and_clean_data_simple(dist, n_samples, batch_size, dtype, seed)
    else:
        return generate_and_clean_data_simple(dist, n_samples, batch_size, dtype, seed)