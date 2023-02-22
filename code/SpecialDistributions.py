
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
import matplotlib.lines as mlines
import corner
import matplotlib.pyplot as plt
tfb= tfp.bijectors


#############TRUNCATED##########################
def TruncatedDistributions(ndims):

    if ndims==8:
        trunc_dist=Truncated8()
    
    if ndims==16:
        trunc_dist=Truncated16()
        


    return trunc_dist
    
def Truncated8():

    joint = tfd.Blockwise(tfd.JointDistributionSequential([
                tfd.TruncatedNormal(0,2,-5,5),
                tfd.TruncatedNormal(1,1,-3.8,3.8),
                tfd.TruncatedNormal( 0.1, 1, 0, 10),
                tfd.TruncatedNormal( -1,.8,-2.2 , 4),
                
                tfd.TruncatedNormal(1,2,-3,4),
                tfd.TruncatedNormal( 2, 1,-100, 3),
                tfd.Mixture(
        cat=tfd.Categorical(probs=[.3,.7]),
        components=[tfd.Normal(loc=-1.8,scale=0.4),
                    tfd.Normal(loc=1,scale=0.2)
    
        ]),
                tfd.TruncatedNormal(0,1,-1,1)
        
                ]))
    print(joint)
    return joint
    
    
def Truncated16():

    #dist9
    uniform=tfd.Mixture(
        cat=tfd.Categorical(probs=[.3,.7]),
        components=[tfd.Normal(loc=-3,scale=1),
                    tfd.Normal(loc=2.8,scale=1.2)
    
        ])
    chain = tfb.Chain([tfb.SoftClip(low=-4.5, high=5,hinge_softness=.5)])
    dist9=tfd.TransformedDistribution(uniform,chain)
    
    #dist13
    uniform=tfd.Mixture(
    cat=tfd.Categorical(probs=[.2,.6,.2]),
    components=[tfd.Normal(loc=-3.7,scale=.5),
                    tfd.Normal(loc=0,scale=1),
                    tfd.Normal(loc=3.8,scale=.5),
    
        ])
    chain = tfb.Chain([tfb.SoftClip(low=-5.5, high=4.5,hinge_softness=.5)])
    dist13=tfd.TransformedDistribution(uniform,chain)
    
    #dist14
    uniform=tfd.Mixture(
    cat=tfd.Categorical(probs=[.2,.6]),
    components=[tfd.Normal(loc=-3,scale=1),
                    tfd.Normal(loc=2,scale=1),
    
        ])
    chain = tfb.Chain([tfb.SoftClip(low=-5, high=7,hinge_softness=.5)])
    dist14=tfd.TransformedDistribution(uniform,chain)
        
    #dist15
    uniform=tfd.Uniform(-3,3)
    dist15=tfd.TransformedDistribution(uniform,tfb.Sigmoid())
    
    #dist16
    uniform=tfd.TruncatedNormal(1,1,-4,4.5)
    chain = tfb.Chain([tfb.Sigmoid()])
    dist16=tfd.TransformedDistribution(uniform,chain)
        
    joint = tfd.Blockwise(tfd.JointDistributionSequential([
    
                tfd.TruncatedNormal(0,2,-5,5),
                tfd.TruncatedNormal(1,1,-4,4),
                tfd.TruncatedNormal( 0.1, 1, 0, 10),
                tfd.TruncatedNormal( -1,.8,-2.2 , 4),
                
                tfd.TruncatedNormal(1,2,-3,4),
                tfd.TruncatedNormal( 2, 1,-100, 3),
                tfd.Mixture(
        cat=tfd.Categorical(probs=[.3,.7]),
        components=[tfd.Normal(loc=-1.8,scale=0.4),
                    tfd.Normal(loc=1,scale=0.2)
    
        ]),
                tfd.TruncatedNormal(0,1,-1,1),
                dist9,
                tfd.TruncatedNormal(-2, 2,-5,5),
                tfd.TruncatedNormal( 0, .6, -.8,.8 ),
                tfd.TruncatedNormal( -1,.8,-2.2 , 4),
                dist13,
                dist14,
                dist15,
                dist16
                
                
        
                ]))
    print(joint)
    return joint
    
    
#############SOMERESONANCES##########################
def SomeResonanceDistributions(ndims):

    if ndims==8:
        sreson_dist=SomeResonances8()
    
    if ndims==16:
        sreson_dist=SomeResonances16()
        


    return sreson_dist


def SomeResonances8():

    joint = tfd.Blockwise(tfd.JointDistributionSequential([
                tfd.Normal(loc=0,scale=1),
                tfd.Normal(loc=3.8,scale=.5),
                tfd.TruncatedCauchy(0,.001,-.1,.1),
                tfd.TruncatedCauchy(0,.01,-.5,.5),
                tfd.Normal(loc=-1,scale=1.2),
                tfd.TruncatedCauchy(0,.0001,-.01,.01),
                tfd.Mixture(
        cat=tfd.Categorical(probs=[.5,.5]),
        components=[tfd.TruncatedCauchy(-.3,.005,-.7,.7),
                    tfd.TruncatedCauchy(.3,.009,-.7,.7)
    
        ]),
        tfd.Normal(loc=0,scale=1)
          
              
        
                ]))
    print(joint)
    return joint



def SomeResonances16():

    joint = tfd.Blockwise(tfd.JointDistributionSequential([
                tfd.Normal(loc=0,scale=1),
                tfd.Normal(loc=3.8,scale=.5),
                tfd.TruncatedCauchy(0,.001,-.1,.1),
                tfd.TruncatedCauchy(0,.01,-.5,.5),
                tfd.Normal(loc=-1,scale=1.2),
                tfd.TruncatedCauchy(0,.0001,-.01,.01),
                tfd.Mixture(
        cat=tfd.Categorical(probs=[.5,.5]),
        components=[tfd.TruncatedCauchy(-.3,.005,-.7,.7),
                    tfd.TruncatedCauchy(.3,.009,-.7,.7)
    
        ]),
        tfd.Mixture(
        cat=tfd.Categorical(probs=[.5,.5]),
        components=[tfd.TruncatedCauchy(-.2,.0005,-.7,.7),
                    tfd.TruncatedCauchy(.4,.0009,-.7,.7)
    
        ]),
        tfd.Normal(loc=0,scale=1),
                tfd.Normal(loc=0,scale=1),
                tfd.Normal(loc=3.8,scale=.5),
                tfd.TruncatedCauchy(0,.0001,-.01,.01),
                tfd.TruncatedCauchy(0,.01,-.5,.5),
                tfd.Normal(loc=-1,scale=1.2),
                tfd.TruncatedCauchy(0,.000001,-.0001,.0001),
                tfd.Normal(loc=0,scale=1),
              
        
                ]))
 
    return joint
    
    
#############ALLRESONANCES##########################


def AllResonanceDistributions(ndims):

    if ndims==8:
        areson_dist=AllResonances8()
    
    if ndims==16:
        areson_dist=AllResonances16()
        


    return areson_dist


def AllResonances8():



    joint = tfd.Blockwise(tfd.JointDistributionSequential([
                tfd.TruncatedCauchy(0,.01,-.5,.5),
                tfd.TruncatedCauchy(2.02,.0001,2.01,2.03),
                tfd.TruncatedCauchy(0,.001,-.1,.1),
                tfd.TruncatedCauchy(0,.01,-.5,.5),
                tfd.TruncatedCauchy(.3,.009,-.7,.7),
                tfd.TruncatedCauchy(0,.0001,-.01,.01),
                tfd.Mixture(
        cat=tfd.Categorical(probs=[.5,.5]),
        components=[tfd.TruncatedCauchy(-.3,.005,-.7,.7),
                    tfd.TruncatedCauchy(.3,.009,-.7,.7)
    
        ]),
        tfd.Mixture(
        cat=tfd.Categorical(probs=[.5,.5]),
        components=[tfd.TruncatedCauchy(-.2,.0005,-.7,.7),
                    tfd.TruncatedCauchy(.4,.0009,-.7,.7)
    
        ])
        ]))
        
    return joint
        

def AllResonances16():



    joint = tfd.Blockwise(tfd.JointDistributionSequential([
                tfd.TruncatedCauchy(0,.01,-.5,.5),
                tfd.TruncatedCauchy(2.02,.0001,2.01,2.03),
                tfd.TruncatedCauchy(0,.001,-.1,.1),
                tfd.TruncatedCauchy(0,.01,-.5,.5),
                tfd.TruncatedCauchy(.3,.009,-.7,.7),
                tfd.TruncatedCauchy(0,.0001,-.01,.01),
                tfd.Mixture(
        cat=tfd.Categorical(probs=[.5,.5]),
        components=[tfd.TruncatedCauchy(-.3,.005,-.7,.7),
                    tfd.TruncatedCauchy(.3,.009,-.7,.7)
    
        ]),
        tfd.Mixture(
        cat=tfd.Categorical(probs=[.5,.5]),
        components=[tfd.TruncatedCauchy(-.2,.0005,-.7,.7),
                    tfd.TruncatedCauchy(.4,.0009,-.7,.7)
    
        ]),
                tfd.TruncatedCauchy(1,.0001,.99,1.01),
                tfd.TruncatedCauchy(0,.00001,-.001,.001),
                tfd.TruncatedCauchy(3,.00001,2.999,3.001),
                tfd.TruncatedCauchy(0,.0001,-.01,.01),
                tfd.TruncatedCauchy(0,.01,-.5,.5),
                tfd.TruncatedCauchy(0,.01,-.5,.5),
                tfd.TruncatedCauchy(0,.000001,-.0001,.0001),
                tfd.TruncatedCauchy(0,.0001,-.01,.01),
              
        
                ]))

    return joint
