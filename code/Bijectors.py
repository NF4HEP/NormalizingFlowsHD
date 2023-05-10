    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:36:34 2019

@author: reyes-gonzalez
"""
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
from RealNVP import *
#from Cspline import *
from Cspline import *
import MAF_spline
import pickle


def RandomShuffle(ndims):

    arr = np.arange(ndims)
    np.random.shuffle(arr)
    random_shuffle=tf.cast(arr, tf.int32)
    return random_shuffle

def ReverseShuffle(ndims):

    arr = np.arange(ndims)
    arr=np.flip(arr)
    reverse_shuffle=tf.cast(arr, tf.int32)
    return reverse_shuffle


def DecimalToBinary(ndims,n_bijectors):

    binaries_list=[]
    for dec in range(ndims):
        biny= bin(dec).replace("0b", "").zfill(n_bijectors)
        binaries_list.append(biny)
        

    return binaries_list

def Log2D(ndims):
    nlog2d=int(np.log2(ndims))
    if 2**nlog2d==ndims:
        n_bijectors=nlog2d
    else:
        n_bijectors=1+nlog2d

    return n_bijectors

def ShuffleMask(binaries_list,bij):

    mask=[]
    for biny in binaries_list:
        val=int(biny[bij])
        mask.append(val)
    mask=tf.cast(mask,dtype=tf.int32)
    return mask


def GetRemDims(ndims,mask):
    
    rem_dims=int(tf.math.reduce_sum(mask))
    
    return rem_dims

def Shufflefirst(mask,ndims,rem_dims):


    k=0
    permutation=[]
    zeros=[]
    ones=[]
    
    
    for  elem in mask:
    
        if elem==0:
            zeros.append(k)
        if elem==1:
            ones.append(k)
        k=k+1
        
    print(zeros)
    print(ones)
    print(zeros+ones)
    permutation=tf.cast(zeros+ones,dtype=tf.int32)
    return permutation


def ShuffleSecond(ndims,rem_dims):

    order=np.concatenate((np.arange(ndims-rem_dims,ndims),np.arange(0,ndims-rem_dims)))
    permutation=tf.cast(order,dtype=tf.int32)
    return permutation






def MAF(ndims,hidden_units,activation, hidden_degrees='equal',
use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, kernel_constraint=None, bias_constraint=None,name='MAF'):

    
    made=tfb.AutoregressiveNetwork(params=2,event_shape=[ndims],hidden_units=hidden_units,activation=activation,hidden_degrees=hidden_degrees, use_bias=use_bias, kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
    
    return tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)
    


def MAFN(ndims,num_bijectors,hidden_layers,activation,hidden_degrees='equal',use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, kernel_constraint=None, bias_constraint=None,perm_style='bi-partition',shuffle='Noshufffle'):
   
    maf=MAF(ndims,hidden_layers,activation,hidden_degrees=hidden_degrees, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
   
    if perm_style=='bi-partition':
   
        permutation=tf.cast(np.concatenate((np.arange(int(ndims/2),ndims),np.arange(0,int(ndims/2)))), tf.int32)
    if perm_style=='reverse':
        
        permutation=ReverseShuffle(ndims)
    
    
    if shuffle=='Noshuffle':
        bijectors=[]
        for _ in range(num_bijectors):
            #bijectors.append(tfb.BatchNormalization())

            bijectors.append(maf)
        
            bijectors.append(tfb.Permute(permutation=permutation))
        
    
        #-1 means to reject last permutation
        flow_bijector=tfb.Chain(list(reversed(bijectors[:-1])))
        
    if shuffle=='RandomShuffle':
        bijectors=[]
        for _ in range(num_bijectors):
            #bijectors.append(tfb.BatchNormalization())

            bijectors.append(maf)
            bijectors.append(tfb.Permute(permutation=permutation))
            bijectors.append(maf)
            bijectors.append(tfb.Permute(permutation=RandomShuffle(ndims)))
    
        #-1 means to reject last permutation
        print(bijectors)
        print(permutation)
        flow_bijector=tfb.Chain(list(reversed(bijectors[:-1])))
            
    
    return flow_bijector
    





def RealNVPN(ndims,rem_dims,num_bijectors,hidden_layers,activation,use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None,perm_style='bi-partition',shuffle='Noshuffle'):

    
    
    if perm_style=='bi-partition':
   
        permutation=tf.cast(np.concatenate((np.arange(int(ndims/2),ndims),np.arange(0,int(ndims/2)))), tf.int32)
    if perm_style=='reverse':
        
        permutation=ReverseShuffle(ndims)
    
    if shuffle=='Noshuffle':
        bijectors=[]

        for i in range(num_bijectors):
            #bijectors.append(tfb.BatchNormalization())
            bijectors.append(RealNVP(ndims,rem_dims,hidden_layers,activation,use_bias,
            kernel_initializer,
            bias_initializer, kernel_regularizer,
            bias_regularizer, activity_regularizer, kernel_constraint,
            bias_constraint))
            bijectors.append(tfp.bijectors.Permute(permutation))
        bijector = tfb.Chain(bijectors=list(reversed(bijectors[:-1])), name='chain_of_real_nvp')
        
    elif shuffle=='RandomShuffle':
        bijectors=[]


        for i in range(num_bijectors):
            #bijectors.append(tfb.BatchNormalization())
            bijectors.append(RealNVP(ndims,rem_dims,hidden_layers,activation,use_bias,
            kernel_initializer,
            bias_initializer, kernel_regularizer,
            bias_regularizer, activity_regularizer, kernel_constraint,
            bias_constraint))
            ###permute transformed and transforming dims
            bijectors.append(tfp.bijectors.Permute(permutation))
            
            bijectors.append(RealNVP(ndims,rem_dims,hidden_layers,activation,use_bias,
            kernel_initializer,
            bias_initializer, kernel_regularizer,
            bias_regularizer, activity_regularizer, kernel_constraint,
            bias_constraint))
            ##shuffle
            bijectors.append(tfp.bijectors.Permute(RandomShuffle(ndims)))
            

        bijector = tfb.Chain(bijectors=list(reversed(bijectors[:-1])), name='chain_of_real_nvp')

        
    elif 'BinaryWiseShuffle':
        print(perm_style)
        print(shuffle)
        n_bijectors=Log2D(ndims)
        binaries_list=DecimalToBinary(ndims,n_bijectors)
        print(binaries_list)
        bijectors=[]
        for bij in range(n_bijectors):
            mask=ShuffleMask(binaries_list,bij)
            rem_dims=GetRemDims(ndims,mask)
            
            
            BinShufflepermutation=Shufflefirst(mask,ndims,rem_dims)
            bijectors.append(tfp.bijectors.Permute(BinShufflepermutation))
            
            
            bijectors.append(RealNVP(ndims,rem_dims,hidden_layers,activation,use_bias,
            kernel_initializer,
            bias_initializer, kernel_regularizer,
            bias_regularizer, activity_regularizer, kernel_constraint,
            bias_constraint))
            
            bi_partiion_permutation=ShuffleSecond(ndims,rem_dims)
            bijectors.append(tfp.bijectors.Permute(bi_partiion_permutation))
            
            bijectors.append(RealNVP(ndims,rem_dims,hidden_layers,activation,use_bias,
            kernel_initializer,
            bias_initializer, kernel_regularizer,
            bias_regularizer, activity_regularizer, kernel_constraint,
            bias_constraint))
        print(bijectors)
        bijector = tfb.Chain(bijectors=list(reversed(bijectors[:-1])), name='chain_of_real_nvp')
        
        
    
    
    return bijector
    


def CsplineN(ndims,rem_dims,spline_knots,n_bijectors,range_min,n_hidden=[128,128,128],activation='relu',use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None,perm_style='bi-partition',shuffle='Noshuffle'):

    if perm_style=='bi-partition':
   
        permutation=tf.cast(np.concatenate((np.arange(int(ndims/2),ndims),np.arange(0,int(ndims/2)))), tf.int32)
    if perm_style=='reverse':
        
        permutation=ReverseShuffle(ndims)

    if shuffle=='Noshuffle':
        bijectors=[]

        #bijectors.append(tfb.BatchNormalization())
    
    
        for i in range(n_bijectors):
            #bijectors.append(tfb.BatchNormalization())
            bijectors.append(Cspline(ndims,rem_dims,spline_knots,range_min,n_hidden,activation,use_bias,
        kernel_initializer,
        bias_initializer, kernel_regularizer,
        bias_regularizer, activity_regularizer, kernel_constraint,
        bias_constraint,name=Cspline))
            bijectors.append(tfp.bijectors.Permute(permutation))
        bijector = tfb.Chain(bijectors=list(reversed(bijectors[:-1])))
        
        
    if shuffle=='RandomShuffle':
        bijectors=[]
 
        #bijectors.append(tfb.BatchNormalization())
    
    
        for i in range(n_bijectors):
            #bijectors.append(tfb.BatchNormalization())
            bijectors.append(Cspline(ndims,rem_dims,spline_knots,range_min,n_hidden,activation,use_bias,
        kernel_initializer,
        bias_initializer, kernel_regularizer,
        bias_regularizer, activity_regularizer, kernel_constraint,
        bias_constraint,name=Cspline))
            ###permute transformed and transforming dims
            bijectors.append(tfp.bijectors.Permute(permutation))
            bijectors.append(Cspline(ndims,rem_dims,spline_knots,range_min,n_hidden,activation,use_bias,
        kernel_initializer,
        bias_initializer, kernel_regularizer,
        bias_regularizer, activity_regularizer, kernel_constraint,
        bias_constraint,name=Cspline))
            bijectors.append(tfp.bijectors.Permute(RandomShuffle(ndims)))
            
        
            
            
        print(bijectors)
        bijector = tfb.Chain(bijectors=list(reversed(bijectors[:-1])))
        
        
    elif shuffle=='BinaryWiseShuffle':
        print(perm_style)
        print(shuffle)
        n_bijectors=Log2D(ndims)
        binaries_list=DecimalToBinary(ndims,n_bijectors)
        print(binaries_list)
        bijectors=[]
        for bij in range(n_bijectors):
            mask=ShuffleMask(binaries_list,bij)
            rem_dims=GetRemDims(ndims,mask)
            
            
            BinShufflepermutation=Shufflefirst(mask,ndims,rem_dims)
            bijectors.append(tfp.bijectors.Permute(BinShufflepermutation))
            
            
            bijectors.append(Cspline(ndims,rem_dims,spline_knots,range_min,n_hidden,activation,use_bias,
        kernel_initializer,
        bias_initializer, kernel_regularizer,
        bias_regularizer, activity_regularizer, kernel_constraint,
        bias_constraint,name=Cspline))
            
            bi_partiion_permutation=ShuffleSecond(ndims,rem_dims)
            bijectors.append(tfp.bijectors.Permute(bi_partiion_permutation))
            
            bijectors.append(Cspline(ndims,rem_dims,spline_knots,range_min,n_hidden,activation,use_bias,
        kernel_initializer,
        bias_initializer, kernel_regularizer,
        bias_regularizer, activity_regularizer, kernel_constraint,
        bias_constraint,name=Cspline))
        print(bijectors)
        bijector = tfb.Chain(bijectors=list(reversed(bijectors[:-1])), name='chain_of_real_nvp')


    return bijector


def MAFspline(ndims,n_hidden,activation,spline_knots,range_min,hidden_degrees='equal',use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, kernel_constraint=None, bias_constraint=None):

    
    ann=tfb.AutoregressiveNetwork(3*spline_knots-1,hidden_units=n_hidden,activation=activation,hidden_degrees=hidden_degrees, use_bias=use_bias, kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
    
    return MAF_spline.MaskedAutoregressiveFlow(shift_and_log_scale_fn=ann,spline_knots=spline_knots,range_min=range_min,name='MAFspline')
    


def MAFNspline(ndims,spline_knots,num_bijectors,range_min,n_hidden=[128,128,128],activation='relu',hidden_degrees='equal',use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, kernel_constraint=None, bias_constraint=None,perm_style='bi-partition',shuffle='Noshuffle'):
   
   
    mafspline=MAFspline(ndims,n_hidden,activation,spline_knots,range_min,hidden_degrees=hidden_degrees, use_bias=use_bias, kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
    
    
    if perm_style=='bi-partition':
   
        permutation=tf.cast(np.concatenate((np.arange(int(ndims/2),ndims),np.arange(0,int(ndims/2)))), tf.int32)
    if perm_style=='reverse':
        
        permutation=ReverseShuffle(ndims)

    if shuffle=='Noshuffle':
        bijectors=[]
        for _ in range(num_bijectors):
            #bijectors.append(tfb.BatchNormalization())
       
            bijectors.append(mafspline)
            bijectors.append(tfb.Permute(permutation=permutation))
            flow_bijector=tfb.Chain(list(reversed(bijectors[:-1])))
    if shuffle=='RandomShuffle':
        bijectors=[]
        for _ in range(num_bijectors):
            #bijectors.append(tfb.BatchNormalization())
       
            bijectors.append(mafspline)
            bijectors.append(tfb.Permute(permutation=permutation))
            bijectors.append(mafspline)
            bijectors.append(tfb.Permute(permutation=RandomShuffle(ndims)))
            
    
        #-1 means to reject last permutation
        print(bijectors)
        flow_bijector=tfb.Chain(list(reversed(bijectors[:-1])))
    
    return flow_bijector


def postprocess_data(data,preprocess_params):

    means=preprocess_params.get('means')
    stds=preprocess_params.get('stds')

    postprocess_data=data*stds+means

    return postprocess_data


def load_preprocess_params(preporcess_params_path):
    with open(preporcess_params_path,'rb') as file:
        preprocess_params=pickle.load(file)
    
    means=preprocess_params.get('means')
    stds=preprocess_params.get('stds')

    return means,stds


def MAFNsplinePreprocess(ndims,spline_knots,num_bijectors,range_min,n_hidden=[128,128,128],activation='relu',hidden_degrees='equal',use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, kernel_constraint=None, bias_constraint=None):

    permutation=tf.cast(np.concatenate((np.arange(int(ndims/2),ndims),np.arange(0,int(ndims/2)))), tf.int32)
    bijectors=[]




    for _ in range(num_bijectors):
        #bijectors.append(tfb.BatchNormalization())
        mafspline=MAFspline(ndims,n_hidden,activation,spline_knots,range_min,hidden_degrees=hidden_degrees, use_bias=use_bias, kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        bijectors.append(mafspline)
        if _ < num_bijectors-1:
            bijectors.append(tfb.Permute(permutation=permutation))


    preporcess_params_path='preprocess_data_flavorfit.pcl'
    means,stds=load_preprocess_params(preporcess_params_path)
    
    means=tf.cast(means,dtype=tf.float32)
    stds=tf.cast(stds,dtype=tf.float32)

    pre_scale=tfb.Scale(scale=stds)
    pre_shift=tfb.Shift(shift=means)
    
    preprocess=tfb.Chain([pre_shift,pre_scale])
    bijectors.append(preprocess)
    #-1 means to reject last permutation
    flow_bijector=tfb.Chain(list(reversed(bijectors[:])))

    return flow_bijector





def MAFNsplineClip(ndims,spline_knots,num_bijectors,range_min,n_hidden=[128,128,128],activation='relu',hidden_degrees='equal',use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, kernel_constraint=None, bias_constraint=None):

    permutation=tf.cast(np.concatenate((np.arange(int(ndims/2),ndims),np.arange(0,int(ndims/2)))), tf.int32)
    bijectors=[]

    for _ in range(num_bijectors):
        #bijectors.append(tfb.BatchNormalization())
        mafspline=MAFspline(ndims,n_hidden,activation,spline_knots,range_min,hidden_degrees=hidden_degrees, use_bias=use_bias, kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        bijectors.append(mafspline)
        bijectors.append(tfb.Permute(permutation=permutation))

    bijectors.append(tfb.SoftClip(low=.2, high=1, hinge_softness=1))
    #-1 means to reject last permutation
    flow_bijector=tfb.Chain(list(reversed(bijectors[:-1])))

    return flow_bijector


def ChooseBijector(bijector_name,ndims,spline_knots,nbijectors,range_min,hidden_layers,activation,regulariser,eps_regulariser,perm_style='bi-partition',shuffle='Noshuffle'):
    if regulariser=='l1':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
    if regulariser=='l2':
        regulariser=tf.keras.regularizers.l2(eps_regulariser)
    else:
        regulariser=None
    if bijector_name=='CsplineN':
        rem_dims=int(ndims/2)
        bijector=CsplineN(ndims,rem_dims,spline_knots,nbijectors,range_min,hidden_layers,activation,kernel_regularizer=regulariser,perm_style=perm_style,shuffle=shuffle)
    elif bijector_name=='MsplineN':
        #regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=MAFNspline(ndims,spline_knots,nbijectors,range_min,hidden_layers,activation,kernel_initializer='glorot_uniform',kernel_regularizer=regulariser,perm_style=perm_style,shuffle=shuffle)
    elif bijector_name=='MAFN':
        #regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=MAFN(ndims,nbijectors,hidden_layers,activation,kernel_regularizer=regulariser,perm_style=perm_style,shuffle=shuffle)
    elif bijector_name=='RealNVPN':
        rem_dims=int(ndims/2)
        #regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=RealNVPN(ndims,rem_dims,nbijectors,hidden_layers,activation,kernel_regularizer=regulariser,perm_style=perm_style,shuffle=shuffle)
    else:
        raise Exception("Bijector name not supported.")
    return bijector
