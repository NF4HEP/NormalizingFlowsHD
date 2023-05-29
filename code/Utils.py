import os
import numpy as np
import codecs
import random
import json
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb= tfp.bijectors
from tensorflow.keras.layers import Input # type" ignore
from tensorflow.keras import Model # type: ignore
import pandas as pd

import MixtureDistributions

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def ResultsToDict(results_dict,run_number,seed_train,seed_test,ndims,nsamples,corr,bijector_name,nbijectors,activation,spline_knots,range_min,ks_mean,ks_std,ks_list,ad_mean,ad_std,ad_list,wd_mean,wd_std,wd_list,swd_mean,swd_std,swd_list,fn_mean,fn_std,fn_list,hidden_layers,batch_size,eps_regulariser,regulariser,epochs_input,epochs_output,training_time,prediction_time,training_device):
    """
    Function that writes results to the a dictionary.
    """
    results_dict.get('run_n').append(run_number)
    results_dict.get('seed_train').append(seed_train)
    results_dict.get('seed_test').append(seed_test)
    results_dict.get('ndims').append(ndims)
    results_dict.get('nsamples').append(nsamples)
    results_dict.get('correlation').append(corr)
    results_dict.get('nbijectors').append(nbijectors)
    results_dict.get('bijector').append(bijector_name)
    results_dict.get('activation').append(activation)
    results_dict.get('spline_knots').append(spline_knots)
    results_dict.get('range_min').append(range_min)
    results_dict.get('eps_regulariser').append(eps_regulariser)
    results_dict.get('regulariser').append(regulariser)
    results_dict.get('ks_mean').append(ks_mean)
    results_dict.get('ks_std').append(ks_std)
    results_dict.get('ks_list').append(ks_list)
    results_dict.get('ad_mean').append(ad_mean)
    results_dict.get('ad_std').append(ad_std)
    results_dict.get('ad_list').append(ad_list)
    results_dict.get('wd_mean').append(wd_mean)
    results_dict.get('wd_std').append(wd_std)
    results_dict.get('wd_list').append(wd_list)
    results_dict.get('swd_mean').append(swd_mean)
    results_dict.get('swd_std').append(swd_std)
    results_dict.get('swd_list').append(swd_list)
    results_dict.get('fn_mean').append(fn_mean)
    results_dict.get('fn_std').append(fn_std)
    results_dict.get('fn_list').append(fn_list)
    results_dict.get('epochs_input').append(epochs_input)
    results_dict.get('epochs_output').append(epochs_output)
    results_dict.get('training_time').append(training_time)
    results_dict.get('hidden_layers').append(hidden_layers)
    results_dict.get('batch_size').append(batch_size)
    results_dict.get('prediction_time').append(training_device)
    results_dict.get('training_device').append(training_device)
    return results_dict

def logger(log_file_name,results_dict):
    """
    Logger that writes results of each run to a common log file.
    """
    dict_copy = results_dict.copy()
    dict_copy.pop('ks_list')
    dict_copy.pop('ad_list')
    dict_copy.pop('wd_list')
    dict_copy.pop('swd_list')
    dict_copy.pop('fn_list')
    log_file=open(log_file_name,'a')
    string_list=[]
    for key in dict_copy.keys():
        string_list.append(str(dict_copy.get(key)[-1]))
    string=','.join(string_list)
    log_file.write(string)
    log_file.write('\n')
    log_file.close()
    return

#def logger_nan(run_number):
#    """
#    Logger that takes care of nan runs.
#    """
#    log_file=open(log_file_name,'a')
#    log_file.write(str(run_number)+",")
#    log_file.write('\n')
#    log_file.close()
#    return

def results_current(path_to_results,results_dict):
    """
    Function that writes results of the current run to the results.txt file.
    """
    dict_copy = results_dict.copy()
    dict_copy.pop('ks_list')
    dict_copy.pop('ad_list')
    dict_copy.pop('wd_list')
    dict_copy.pop('swd_list')
    dict_copy.pop('fn_list')
    currrent_results_file=open(path_to_results+'results.txt','w')
    header=','.join(list(dict_copy.keys()))
    currrent_results_file.write(header)
    currrent_results_file.write('\n')
    string_list=[]
    for key in dict_copy.keys():
        string_list.append(str(dict_copy.get(key)[-1]))
    string=','.join(string_list)
    currrent_results_file.write(string)
    currrent_results_file.write('\n')
    currrent_results_file.close()
    return

def save_hyperparams(path_to_results,hyperparams_dict,run_number,seed_train,seed_test,ndims,nsamples,corr,bijector_name,nbijectors,spline_knots,range_min,hllabel,batch_size,activation,eps_regulariser,regulariser,dist_seed,test_seed,training_device):
    """
    Function that writes hyperparameters values to a dictionary and saves it to the hyperparam.txt file.
    """
    hyperparams_dict.get('run_n').append(run_number)
    hyperparams_dict.get('seed_train').append(seed_train)
    hyperparams_dict.get('seed_test').append(seed_test)
    hyperparams_dict.get('ndims').append(ndims)
    hyperparams_dict.get('nsamples').append(nsamples)
    hyperparams_dict.get('correlation').append(corr)
    hyperparams_dict.get('bijector').append(bijector_name)
    hyperparams_dict.get('nbijectors').append(nbijectors)
    hyperparams_dict.get('spline_knots').append(spline_knots)
    hyperparams_dict.get('range_min').append(range_min)
    hyperparams_dict.get('hidden_layers').append(hllabel)
    hyperparams_dict.get('batch_size').append(batch_size)
    hyperparams_dict.get('activation').append(activation)
    hyperparams_dict.get('eps_regulariser').append(eps_regulariser)
    hyperparams_dict.get('regulariser').append(regulariser)
    hyperparams_dict.get('dist_seed').append(dist_seed)
    hyperparams_dict.get('test_seed').append(test_seed)
    hyperparams_dict.get('training_device').append(training_device)
    hyperparams_frame=pd.DataFrame(hyperparams_dict)
    hyperparams_frame.to_csv(path_to_results+'hyperparams.txt',index=False)
    return hyperparams_dict

def load_model(nf_dist,path_to_results,ndims,lr=.00001):
    """
    Function that loads a model by recreating it, recompiling it and loading checkpointed weights.
    """
    x_ = Input(shape=(ndims,), dtype=tf.float32)
    log_prob_ = nf_dist.log_prob(x_)
    model = Model(x_, log_prob_)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                  loss=lambda _, log_prob: -log_prob)
    model.load_weights(path_to_results+'model_checkpoint/weights')
    return nf_dist,model

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

def save_bijector_info(bijector,path_to_results):
    """
    Function that saves the bijecor.
    """
    bij_out_file=open(path_to_results+'bijector_chain.txt','w')
    for bij in list(bijector.bijectors):
        bij_out_file.write(bij.name)
        bij_out_file.write('\n')
    bij_out_file.close()
    return

@tf.function
def nf_sample_iter(nf_dist,iter_size,n_iters,seed=0):
    """
    To be decumented.
    """
    reset_random_seeds(seed)
    #first iter
    sample_all=nf_dist.sample(iter_size,seed=seed)
    for j in range(1,n_iters):
        #if j%100==0:
            #print(tf.shape(sample_all))
        sample=nf_dist.sample(iter_size,seed=seed)
        #sample=postprocess_data(sample,preprocess_params)
        sample_all=tf.concat([sample_all,sample],0)
        #if j%1==0:
        #    with open(path_to_results+'nf_sample_5_'+str(j)+'.npy', 'wb') as f:
        #        np.save(f, sample, allow_pickle=True)
        #tf.keras.backend.clear_session()
    return sample_all

def sample_save(test_dist,nf_dist,path_to_results,sample_size=100000,iter_size=10000,rot=None,seed=0):
    """
    Function that saves the samples.
    """
    print('saving samples...')
    n_iters=int(sample_size/iter_size)
    sample_all=nf_sample_iter(nf_dist,iter_size,2*n_iters,seed=seed)
    sample_all=sample_all.numpy() # type: ignore
    # Check/remove nans
    sample_all_no_nans = sample_all[~np.isnan(sample_all).any(axis=1), :]
    if len(sample_all) != len(sample_all_no_nans):
        print("Samples containing nan have been removed. The fraction of nans over the total samples was:", str((len(sample_all)-len(sample_all_no_nans))/len(sample_all)),".")
    else:
        pass
    sample_all = sample_all_no_nans[:sample_size]
    if rot is not None:
        sample_all = MixtureDistributions.inverse_transform_data(sample_all,rot)
    with open(path_to_results+'nf_sample.npy', 'wb') as f:
        np.save(f, sample_all, allow_pickle=True)
    #with open(path_to_results+'test_sample.npy', 'wb') as f:
    #    np.save(f, test_dist, allow_pickle=True)
    print('samples saved')
    return [test_dist,sample_all]

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
    dict_copy.pop('ks_list')
    dict_copy.pop('ad_list')
    dict_copy.pop('wd_list')
    dict_copy.pop('swd_list')
    dict_copy.pop('fn_list')
    log_file_name=mother_output_dir+'log_file_eternal.txt'
    if os.path.isfile(log_file_name)==False:
        log_file=open(log_file_name,'w')
        header=','.join(list(dict_copy.keys()))
        print(header)
        log_file.write(header)
        log_file.write('\n')
        log_file.close()
    return log_file_name