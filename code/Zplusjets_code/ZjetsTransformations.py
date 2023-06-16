import h5py
import numpy as np
import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt








def Preprocess_jet1(spher_all):
    spher_all_new=spher_all

    #spher_all[:,0]=np.log(spher_all[:,0]-np.min(spher_all[:,0])+1e-1)
    #spher_all[:,3+0]=np.log(spher_all[:,3+0]-np.min(spher_all[:,3+0])+1e-1)
    #spher_all[:,6+0]=np.log(spher_all[:,6+0]-np.min(spher_all[:,6+0])+1e-1)
    spher_all_new[:,0]=np.log(spher_all[:,0])
    spher_all_new[:,3+0]=np.log(spher_all[:,3+0])
    spher_all_new[:,6+0]=np.log(spher_all[:,6+0])


    spher_all_new[:,2]=np.arctanh(spher_all[:,2]/np.pi)
    spher_all_new[:,3+2]=np.arctanh(spher_all[:,3+2]/np.pi)
    spher_all[:,6+2]=np.arctanh(spher_all[:,6+2]/np.pi)

    spher_all_new[:,9]=np.log(spher_all[:,9]+1)


    return spher_all_new
    
    
def Undo_Preprocess_jet1(spher_all):


    #spher_all[:,0]=np.exp(spher_all[:,0])+min_0-1e-1
    #spher_all[:,3]=np.exp(spher_all[:,3])+min_3-1e-1
    #spher_all[:,6]=np.exp(spher_all[:,6])+min_6-1e-1
    
    spher_all[:,0]=np.exp(spher_all[:,0])
    spher_all[:,3]=np.exp(spher_all[:,3])
    spher_all[:,6]=np.exp(spher_all[:,6])
    
    
    
    
    
    spher_all[:,2]=np.tanh(spher_all[:,2])*np.pi
    spher_all[:,3+2]=np.tanh(spher_all[:,3+2])*np.pi
    spher_all[:,6+2]=np.tanh(spher_all[:,6+2])*np.pi

    spher_all[:,9]=np.exp(spher_all[:,9])-1

    return spher_all
    
    

def Preprocess_jet2(spher_all):


    #spher_all[:,0]=np.log(spher_all[:,0]-np.min(spher_all[:,0])+1e-1)
    #spher_all[:,3+0]=np.log(spher_all[:,3+0]-np.min(spher_all[:,3+0])+1e-1)
    #spher_all[:,6+0]=np.log(spher_all[:,6+0]-np.min(spher_all[:,6+0])+1e-1)
    spher_all[:,0]=np.log(spher_all[:,0])
    spher_all[:,3+0]=np.log(spher_all[:,3+0])
    spher_all[:,6+0]=np.log(spher_all[:,6+0])
    spher_all[:,10+0]=np.log(spher_all[:,10+0])


    spher_all[:,2]=np.arctanh(spher_all[:,2]/np.pi)
    spher_all[:,3+2]=np.arctanh(spher_all[:,3+2]/np.pi)
    spher_all[:,6+2]=np.arctanh(spher_all[:,6+2]/np.pi)
    spher_all[:,10+2]=np.arctanh(spher_all[:,10+2]/np.pi)

    spher_all[:,9]=np.log(spher_all[:,9]+1)
    spher_all[:,4+9]=np.log(spher_all[:,4+9]+1)


    return spher_all
    
    
def Undo_Preprocess_jet2(spher_all):


    #spher_all[:,0]=np.exp(spher_all[:,0])+min_0-1e-1
    #spher_all[:,3]=np.exp(spher_all[:,3])+min_3-1e-1
    #spher_all[:,6]=np.exp(spher_all[:,6])+min_6-1e-1
    
    spher_all[:,0]=np.exp(spher_all[:,0])
    spher_all[:,3]=np.exp(spher_all[:,3])
    spher_all[:,6]=np.exp(spher_all[:,6])
    spher_all[:,10]=np.exp(spher_all[:,10])
    
    
    
    
    
    spher_all[:,2]=np.tanh(spher_all[:,2])*np.pi
    spher_all[:,3+2]=np.tanh(spher_all[:,3+2])*np.pi
    spher_all[:,6+2]=np.tanh(spher_all[:,6+2])*np.pi
    spher_all[:,10+2]=np.tanh(spher_all[:,10+2])*np.pi

    spher_all[:,9]=np.exp(spher_all[:,9])-1
    spher_all[:,4+9]=np.exp(spher_all[:,4+9])-1


    return spher_all


def Preprocess_jet3(spher_all):


    #spher_all[:,0]=np.log(spher_all[:,0]-np.min(spher_all[:,0])+1e-1)
    #spher_all[:,3+0]=np.log(spher_all[:,3+0]-np.min(spher_all[:,3+0])+1e-1)
    #spher_all[:,6+0]=np.log(spher_all[:,6+0]-np.min(spher_all[:,6+0])+1e-1)
    spher_all[:,0]=np.log(spher_all[:,0])
    spher_all[:,3+0]=np.log(spher_all[:,3+0])
    spher_all[:,6+0]=np.log(spher_all[:,6+0])
    spher_all[:,10+0]=np.log(spher_all[:,10+0])
    spher_all[:,14+0]=np.log(spher_all[:,14+0])


    spher_all[:,2]=np.arctanh(spher_all[:,2]/np.pi)
    spher_all[:,3+2]=np.arctanh(spher_all[:,3+2]/np.pi)
    spher_all[:,6+2]=np.arctanh(spher_all[:,6+2]/np.pi)
    spher_all[:,10+2]=np.arctanh(spher_all[:,10+2]/np.pi)
    spher_all[:,14+2]=np.arctanh(spher_all[:,14+2]/np.pi)

    spher_all[:,9]=np.log(spher_all[:,9]+1)
    spher_all[:,4+9]=np.log(spher_all[:,4+9]+1)
    spher_all[:,8+9]=np.log(spher_all[:,8+9]+1)


    return spher_all
    
    
def Undo_Preprocess_jet3(spher_all):


    #spher_all[:,0]=np.exp(spher_all[:,0])+min_0-1e-1
    #spher_all[:,3]=np.exp(spher_all[:,3])+min_3-1e-1
    #spher_all[:,6]=np.exp(spher_all[:,6])+min_6-1e-1
    
    spher_all[:,0]=np.exp(spher_all[:,0])
    spher_all[:,3]=np.exp(spher_all[:,3])
    spher_all[:,6]=np.exp(spher_all[:,6])
    spher_all[:,10]=np.exp(spher_all[:,10])
    spher_all[:,14]=np.exp(spher_all[:,14])
    
    
    
    
    spher_all[:,2]=np.tanh(spher_all[:,2])*np.pi
    spher_all[:,3+2]=np.tanh(spher_all[:,3+2])*np.pi
    spher_all[:,6+2]=np.tanh(spher_all[:,6+2])*np.pi
    spher_all[:,10+2]=np.tanh(spher_all[:,10+2])*np.pi
    spher_all[:,14+2]=np.tanh(spher_all[:,14+2])*np.pi
    
    

    spher_all[:,9]=np.exp(spher_all[:,9])-1
    spher_all[:,4+9]=np.exp(spher_all[:,4+9])-1
    spher_all[:,8+9]=np.exp(spher_all[:,8+9])-1

    return spher_all
