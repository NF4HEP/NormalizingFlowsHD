
###Function required for truncated distributions
##It soft clips the NF distribution, to be used after training and before evaluation

def SoftClipTransform(nf_dist,ndims,hinge=1e-5):
    if ndims==16:
        max_conditions=[5,4,10,4,4,3,10000,1,5,5,.8,4,4.5,7,0.9525741,0.9890126]
        min_conditions=[-5,-4,0,-2.2,-3,-100,-9999,-1,-4.5,-5,-.8,-2.2,-5.5,-5,4.7425866e-02,1.8393755e-02]
        
    if ndims==8:
        max_conditions=[5,4,10,4,4,3,10000,1]
        min_conditions=[-5,-4,0,-2.2,-3,-100,-9999,-1]
 
    bijector=tfb.SoftClip(low=min_conditions, high=max_conditions,hinge_softness=hinge)

    nf_dist=nf_dist=tfd.TransformedDistribution(nf_dist,bijector)

    return nf_dist

##Preprocessing functions for resonance distributions

def load_preprocess_params(preporcess_params_path):
    with open(preporcess_params_path,'rb') as file:
        preprocess_params=pickle.load(file)
    
    return preprocess_params


def preprocess_data(data,preprocess_params):
    
    means=preprocess_params.get('means')
    stds=preprocess_params.get('stds')
    
    preprocess_data=(data-means)/stds
    
    return preprocess_data


def postprocess_data(data,preprocess_params):

    means=preprocess_params.get('means')
    stds=preprocess_params.get('stds')
    
    postprocess_data=data*stds+means
    
    return postprocess_data



