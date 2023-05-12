
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle as pkl
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb= tfp.bijectors
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.stats import epps_singleton_2samp
from scipy.stats import anderson_ksamp
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import pandas as pd
import Distributions,Bijectors
#import Trainer_2 as Trainer
#import Metrics_old as Metrics
from statistics import mean,median
import matplotlib.lines as mlines
import corner

def sample_plotter(target_test_data,nf_dist,path_to_plots):

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        f.suptitle('target vs nf')
        x=target_test_data
        ax1.plot(x[:,0],targ_dist.prob(x),'.')
        ax1.set_yscale('log')
        ax1.set_title('target')
        y=nf_dist.sample(target_test_data.shape[1])
        ax2.plot(y[:,0],nf_dist.prob(y),'.')
        ax2.set_yscale('log')
        ax2.set_title('nf')
        f.savefig(path_to_plots+'/sample_plot.pdf')
        ax1.cla()
        ax2.cla()
        f.clf()
        
        return
        
        
def train_plotter(t_losses,v_losses,path_to_plots):
    plt.plot(t_losses,label='train')
    plt.plot(v_losses,label='validation')
    plt.legend()
    plt.title('history')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(path_to_plots+'/loss_plot.pdf')
    plt.close()
    return

#def plot_corner(data):
#    ndatasets = len(data)
#    if ndatasets>4:
#        raise Exception("Up to 6 databases can be plot together.")
#    # Check ndims
#    ndims_list = [np.shape(x)[1] for x  in data]
#    if np.all(np.array(ndims_list)==np.full(ndatasets,ndims_list[0])):
#        ndims = np.shape(data[0])[1]
#    else:
#        raise Exception("Datasets have different number of dimensions")
#    blue_line = mlines.Line2D([], [], color='red', label='data1')
#    red_line = mlines.Line2D([], [], color='blue', label='data2')
#    green_line = mlines.Line2D([], [], color='green', label='data3')
#    gray_line = mlines.Line2D([], [], color='gray', label='data4')
#    figure=corner.corner(data[0],color='red', hist_kwargs={"color": 'red', "linewidth": "1.5"}, 
#                              #data_kwargs={"alpha": 0.8}, 
#                              normalize1d=True)
#    if ndatasets>1:
#        corner.corner(data[1],color='blue',hist_kwargs={"color": 'blue', "linewidth": "1.5","linestyle": "dashed"}, 
#                              contour_kwargs={"linestyles": ["dashed"]},
#                              #data_kwargs={"alpha": 0.8}, 
#                              fig=figure,normalize1d=True)
#    if ndatasets>2:
#        corner.corner(data[2],color='green',hist_kwargs={"color": 'green', "linewidth": "1.5","linestyle": "dotted"}, 
#                              contour_kwargs={"linestyles": ["dotted"]},
#                              #data_kwargs={"alpha": 0.8}, 
#                              fig=figure,normalize1d=True)
#    if ndatasets>3:
#        corner.corner(data[3],color='black',hist_kwargs={"color": 'gray', "linewidth": "1.5","linestyle": "dashdot"}, 
#                              contour_kwargs={"linestyles": ["dashdot"]},
#                              #data_kwargs={"alpha": 0.8}, 
#                              fig=figure,normalize1d=True)
#    legend_list = list(np.array([blue_line,red_line,green_line,gray_line])[:ndatasets])
#    plt.legend(handles=legend_list, bbox_to_anchor=(-ndims+1.8, ndims+.3, 1., 0.) ,fontsize='xx-large')
#    #plt.savefig(path_to_plots+'/corner_plot.png',pil_kwargs={'quality':50})
#    plt.show()
#    plt.close()  

def cornerplotter(target_test_data,nf_dist,path_to_plots,ndims,rot=None,norm=False,max_dim=32):
    # Define the two samples (target and nf)
    shape = target_test_data.shape
    target_samples=target_test_data
    if norm==False:
        nf_samples=nf_dist.sample(2*shape[0]).numpy()
        if rot is not None:
            nf_samples = np.dot(nf_samples,np.transpose(rot))
    else:
        nf_samples=nf_dist
    # Check/remove nans
    nf_samples_no_nans = nf_samples[~np.isnan(nf_samples).any(axis=1), :]
    if len(nf_samples) != len(nf_samples_no_nans):
        print("Samples containing nan have been removed. The fraction of nans over the total samples was:", str((len(nf_samples)-len(nf_samples_no_nans))/len(nf_samples)),".")
    else:
        pass
    nf_samples = nf_samples_no_nans[:shape[0]]
    # Define generic labels
    labels = []
    for i in range(shape[1]):
        labels.append(r"$\theta_{%d}$" % i)
        i = i+1
    # Choose dimensions to plot
    thin = int(shape[1]/max_dim)+1
    if thin<=2:
        thin = 1
    # Select samples to plot
    target_samples = target_samples[:,::thin]
    nf_samples = nf_samples[:,::thin]
    # Select labels
    labels = list(np.array(labels)[::thin])

    n_bins = 50
    #red_bins=50
    #density=(np.max(target_samples,axis=0)-np.min(target_samples,axis=0))/red_bins
    #
    #blue_bins=(np.max(nf_samples,axis=0)-np.min(nf_samples,axis=0))/density
    #blue_bins=blue_bins.astype(int).tolist()

    #file = open(path_to_plots+'/samples.pkl', 'wb')
    #pkl.dump(np.array(target_samples), file, protocol=4)
    #pkl.dump(np.array(nf_samples), file, protocol=4)
    #file.close()

    blue_line = mlines.Line2D([], [], color='red', label='target')
    red_line = mlines.Line2D([], [], color='blue', label='NF')
    figure=corner.corner(target_samples,color='red',bins=n_bins,labels=[r"%s" % s for s in labels])
    corner.corner(nf_samples,color='blue',bins=n_bins,fig=figure)
    plt.legend(handles=[blue_line,red_line], bbox_to_anchor=(-ndims+1.8, ndims+.3, 1., 0.) ,fontsize='xx-large')
    plt.savefig(path_to_plots+'/corner_plot.pdf',pil_kwargs={'quality':50})
    plt.close()
    return


def marginal_plot(target_test_data,sample_nf,path_to_plots,ndims):

 
    n_bins=50

    

    if ndims<=4:
    
        fig, axs = plt.subplots(int(ndims/4), 4, tight_layout=True)
    
        for dim in range(ndims):
    
  
            column=int(dim%4)

            axs[column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            axs[column].hist(sample_nf[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
        
        
            x_axis = axs[column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[column].axes.get_yaxis()
            y_axis.set_visible(False)
    
    
    

    elif ndims>=100:
    
        fig, axs = plt.subplots(int(ndims/10), 10, tight_layout=True)
    
        for dim in range(ndims):
    
  
            row=int(dim/10)
            column=int(dim%10)

            axs[row,column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            axs[row,column].hist(sample_nf[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
        
        
            x_axis = axs[row,column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[row,column].axes.get_yaxis()
            y_axis.set_visible(False)

    else:
        
        
        fig, axs = plt.subplots(int(ndims/4), 4, tight_layout=True)
        for dim in range(ndims):
    
            row=int(dim/4)
            column=int(dim%4)

            axs[row,column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            axs[row,column].hist(sample_nf[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
        
        
            x_axis = axs[row,column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[row,column].axes.get_yaxis()
            y_axis.set_visible(False)
        
    fig.savefig(path_to_plots+'/marginal_plot.pdf',dpi=300)
    fig.clf()

    return

    
