

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

def gaussians(ndims):
    gaussian=tfd.Sample(tfd.Normal(loc=0, scale=1,allow_nan_stats=False),
               sample_shape=[ndims])
    return gaussian



def mixture_of_gaussians(ndims):

    ### Mix_gauss will be our target distribution.
    probs=[0.3,.7]
    mix_gauss=tfd.Sample(tfd.Mixture(
        cat=tfd.Categorical(probs=probs),
        components=[tfd.Normal(loc=3.3,scale=0.4),
                    tfd.Normal(loc=1.8,scale=0.2)
    
        ]),sample_shape=[ndims])


    

    return mix_gauss

def two_moons_dataset(nsamples):

    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    X, y = noisy_moons
    X_data = StandardScaler().fit_transform(X)
    
    return X_data


def correlated_gaussians(ndims):

    
    mu=np.random.uniform(-1,1,ndims).tolist()
    nelements=int((ndims*ndims+ndims)/2)
 
        
        
    matrix_elements= np.random.uniform(0,1,nelements).tolist()
    triangular=tfp.math.fill_triangular(matrix_elements,upper=False)

    print(triangular)
    print(mu)
    
    mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=triangular)
    
    return mvn,triangular
    
    
def gaussians4(nsamples):

    rng = np.random.RandomState()

    scale = 4.
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    centers = [(scale * x, scale * y) for x, y in centers]

    dataset = []
    for i in range(nsamples):
        point = rng.randn(2) * 0.5
        idx = rng.randint(4)
        center = centers[idx]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
    dataset = np.array(dataset, dtype="float32")
    dataset /= 1.414
    return dataset
    




