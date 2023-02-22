
import numpy as np
import tensorflow as tf



def DecimalToBinary(ndims,n_bijectors):

    binaries_list=[]
    for dec in range(ndims):
        biny= bin(dec).replace("0b", "").zfill(n_bijectors)
        binaries_list.append(biny)
        

    return binaries_list

def Log2D(ndims):
    nlog2d=np.log2(ndims)
    if 2**nlog2d==ndims:
        n_bijectors=int(np.log2(ndims))
    else:
        n_bijectors=1+int(np.log2(ndims))

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
    
def ReorderBack():

    return



def RandomShuffle(ndims):

    arr = np.arange(ndims)
    np.random.shuffle(arr)
    random_shuffle=tf.cast(arr, tf.int32)
    return random_shuffle


#print(RandomShuffle(ndims))

def ReverseShuffle(ndims):

    arr = np.arange(ndims)
    arr=np.flip(arr)
    reverse_shuffle=tf.cast(arr, tf.int32)
    return reverse_shuffle


ndims=9
biny= bin(ndims).format(12).replace("0b", "")
print(ndims)


print(np.log2(ndims))
print(Log2D(ndims))
print(2**(Log2D(ndims)-1))
n_bijectors=Log2D(ndims)
print(DecimalToBinary(ndims,n_bijectors))
binaries_list=DecimalToBinary(ndims,n_bijectors)
bij=1
mask=ShuffleMask(binaries_list,bij)
print(mask)
rem_dims=GetRemDims(ndims,mask)
print(rem_dims)

permutation=Shufflefirst(mask,ndims,rem_dims)
print(permutation)

permutation=ShuffleSecond(ndims,rem_dims)
print(permutation)



def RealNVPN_log2D():

    n_bijectors=Log2D(ndims)
    binaries_list=DecimalToBinary(ndims,n_bijectors)
    rem_dims=GetRemDims(ndims,mask)
    
    for bij in n_bijectors:
        mask=ShuffleMask(binaries_list,bij)
        
        permutation=Shufflefirst(mask,ndims,rem_dims)
        bijectors.append(RealNVP(ndims,rem_dims,hidden_layers,activation,use_bias,
            kernel_initializer,
            bias_initializer, kernel_regularizer,
            bias_regularizer, activity_regularizer, kernel_constraint,
            bias_constraint))
        permutation=ShuffleSecond(ndims,rem_dims)
        bijectors.append(RealNVP(ndims,rem_dims,hidden_layers,activation,use_bias,
            kernel_initializer,
            bias_initializer, kernel_regularizer,
            bias_regularizer, activity_regularizer, kernel_constraint,
            bias_constraint))
    
    bijector = tfb.Chain(bijectors=list(reversed(bijectors[:-1])), name='chain_of_real_nvp')
    
    return bijector
