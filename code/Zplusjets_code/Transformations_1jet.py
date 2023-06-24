import sys
import h5py
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

sys.path.append('../')
import corner

def marginal_plot(target_test_data,sample_nf,name):

 
    ndims=np.shape(target_test_data)[1]
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
        
        
        fig, axs = plt.subplots(int(ndims/2),2, tight_layout=True)
        for dim in range(ndims):
    
            row=int(dim/2)
            column=int(dim%2)

            axs[row,column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            axs[row,column].hist(sample_nf[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
        
        
            x_axis = axs[row,column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[row,column].axes.get_yaxis()
            y_axis.set_visible(False)
        
    fig.savefig(name,dpi=300)
    fig.clf()

    return



def cornerplotter(data,name):

    check_inf = np.isfinite(data)
    result = data[check_inf]    
    print(np.shape(data))
    print(np.shape(result))

    ndims=np.shape(data)[1]
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

    figure=corner.corner(data,color='red',bins=n_bins)
    #corner.corner(nf_samples,color='blue',bins=n_bins,fig=figure)
    plt.legend(handles=[blue_line], bbox_to_anchor=(-ndims+1.8, ndims+.3, 1., 0.) ,fontsize='xx-large')
    plt.savefig(name,pil_kwargs={'quality':50})
    #try:
    #    make_pdf_from_img(path_to_plots+'/corner_plot.png')
    #except:
    #    pass
    plt.close()
    return


def cornerplotter_comp(data,data_2,name):

    ndims=np.shape(data)[1]
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

    red_line = mlines.Line2D([], [], color='red', label='orig')
    blue_line = mlines.Line2D([], [], color='red', label='undone')

    figure=corner.corner(data,color='red',bins=n_bins)
    corner.corner(data_2,color='blue',bins=n_bins,fig=figure)
    plt.legend(handles=[red_line,blue_line], bbox_to_anchor=(-ndims+1.8, ndims+.3, 1., 0.) ,fontsize='xx-large')
    plt.savefig(name,pil_kwargs={'quality':50})
    #try:
    #    make_pdf_from_img(path_to_plots+'/corner_plot.png')
    #except:
    #    pass
    plt.close()
    return

#events_dataset_path = "../events/events.h5"
events_dataset_path = "/mnt/project_mnt/teo_fs/rtorre/cernbox/git/GitHub/NormalizingFlows/NF4HEP/NormalizingFlowsHD/ZplusJets_hlvar/events/events.h5"

with h5py.File(events_dataset_path, 'r') as hdf:
    # Save the datasets into the file with unique names
    try:
        events = np.array(hdf['Z+1j'][:]).astype(np.float32)
    except:
        raise Exception("Z+1j not found in the dataset.")





def CoordTransforms(events):

    n_events=np.shape(events)[0]

    ###Muons1

    E_mu1=events[:,0]
    px_mu1=events[:,1]
    py_mu1=events[:,2]
    pz_mu1=events[:,3]


    pt_mu1=np.sqrt(px_mu1**2+py_mu1**2)



    rapid_mu1=.5*np.log((E_mu1+pz_mu1)/(E_mu1-pz_mu1))
    #azim_mu1=np.arccos(pz_mu1/(np.sqrt(px_mu1**2+py_mu1**2+pz_mu1**2)))/np.pi
    #azim_mu1=np.tan(np.sign(py_mu1)*np.arccos(px_mu1/(np.sqrt(px_mu1**2+py_mu1**2)))/np.pi)

    #azim_mu1=np.arctan(py_mu1/px_mu1)
    azim_mu1=np.sign(py_mu1)*np.arccos(px_mu1/(np.sqrt(px_mu1**2+py_mu1**2)))

    #azim_mu1=np.arctanh(azim_mu1/np.pi)
    m_mu1=np.sqrt(E_mu1**2-(px_mu1**2+py_mu1**2+pz_mu1**2))
    mt_mu1=np.sqrt(m_mu1**2+pt_mu1**2)



    #pt_mu1=np.log(pt_mu1-np.min(pt_mu1))
    spher_mu1=np.array([pt_mu1,rapid_mu1,azim_mu1])
    print(spher_mu1)
    print(np.shape(spher_mu1))
    spher_mu1=np.transpose(spher_mu1)
    print(spher_mu1)
    print(np.shape(spher_mu1))


    ###Muons2


    E_mu2=events[:,4+0]
    px_mu2=events[:,4+1]
    py_mu2=events[:,4+2]
    pz_mu2=events[:,4+3]


    pt_mu2=np.sqrt(px_mu2**2+py_mu2**2)
    rapid_mu2=.5*np.log((E_mu2+pz_mu2)/(E_mu2-pz_mu2))

    #azim_mu2=np.arccos(pz_mu2/(np.sqrt(px_mu2**2+py_mu2**2+pz_mu2**2)))/np.pi




    azim_mu2=np.sign(py_mu2)*np.arccos(px_mu2/(np.sqrt(px_mu2**2+py_mu2**2)))

    #m_mu2=np.sqrt(E_mu2**2-(px_mu2**2+py_mu2**2+pz_mu2**2))
    #mt_mu2=np.sqrt(m_mu2**2+pt_mu2**2)

    spher_mu2=np.array([pt_mu2,rapid_mu2,azim_mu2])
    print(spher_mu2)
    print(np.shape(spher_mu2))
    spher_mu2=np.transpose(spher_mu2)
    
    
    print(np.shape(spher_mu2))
    print(spher_mu2)
    print(np.shape(spher_mu2))
    print('max mu2')
    print(np.max(spher_mu2,axis=0))
    print('min mu2')
    print(np.min(spher_mu2,axis=0))
    print('min E-pz')
    print(np.min(E_mu2-pz_mu2,axis=0))

    #indexList = [np.any(i) for i in np.isinf(spher_mu2)]
    #spher_mu2 = np.delete(spher_mu2, indexList, axis=0)
    #name='corner_plot_mu2.png'
    #cornerplotter(spher_mu2,name)
 

    #jets1


    E_jet1=events[:,8+0]
    px_jet1=events[:,8+1]
    py_jet1=events[:,8+2]
    pz_jet1=events[:,8+3]


    pt_jet1=np.sqrt(px_jet1**2+py_jet1**2)
    rapid_jet1=.5*np.log((E_jet1+pz_jet1)/(E_jet1-pz_jet1))
    azim_jet1=np.sign(py_jet1)*np.arccos(px_jet1/(np.sqrt(px_jet1**2+py_jet1**2)))

    m_jet1_sq=E_jet1**2-(px_jet1**2+py_jet1**2+pz_jet1**2)
    print('max mjet1')
    print(np.max(m_jet1_sq,axis=0))
    print('min mjet1')
    print(np.min(m_jet1_sq,axis=0))
 
    
    
    m_jet1=np.sqrt(E_jet1**2-(px_jet1**2+py_jet1**2+pz_jet1**2))
    mt_jet1=np.sqrt(m_jet1**2+pt_jet1**2)

    spher_jet1=np.array([pt_jet1,rapid_jet1,azim_jet1,m_jet1])
    print(spher_jet1)
    print(np.shape(spher_jet1))
    spher_jet1=np.transpose(spher_jet1)
    print(spher_jet1)
    print(np.shape(spher_jet1))


    print('max jet1')
    print(np.max(spher_jet1,axis=0))
    print('min jet1')
    print(np.min(spher_jet1,axis=0))
    print('min E-pz')
    print(np.min(E_jet1-pz_jet1,axis=0))
    #name='corner_plot_jet1.png'
    #cornerplotter(spher_jet1,name)
   
    
    spher_all=np.concatenate((spher_mu1,spher_mu2,spher_jet1),axis=1)
    
    
    
    indexList = [np.any(i) for i in np.isinf(spher_all)]
    spher_all = np.delete(spher_all, indexList, axis=0)
    
    indexList = [np.any(i) for i in np.isnan(spher_all)]
    spher_all = np.delete(spher_all, indexList, axis=0)
    
    #name='corner_plot_all.png'
    #cornerplotter(spher_all,name)
    
    print(spher_all)
    print(np.shape(spher_all))
   
    
    return spher_all



def Preprocess_1(spher_all):


    #spher_all[:,0]=np.log(spher_all[:,0]-np.min(spher_all[:,0])+1e-1)
    #spher_all[:,3+0]=np.log(spher_all[:,3+0]-np.min(spher_all[:,3+0])+1e-1)
    #spher_all[:,6+0]=np.log(spher_all[:,6+0]-np.min(spher_all[:,6+0])+1e-1)
    spher_all[:,0]=np.log(spher_all[:,0])
    spher_all[:,3+0]=np.log(spher_all[:,3+0])
    spher_all[:,6+0]=np.log(spher_all[:,6+0])

    # Needed to avoid rare division by zero in arctanh function
    spher_all[spher_all >=  1] =  0.9999999
    spher_all[spher_all <= -1] = -0.9999999

    spher_all[:,2]=np.arctanh(spher_all[:,2]/np.pi)
    spher_all[:,3+2]=np.arctanh(spher_all[:,3+2]/np.pi)
    spher_all[:,6+2]=np.arctanh(spher_all[:,6+2]/np.pi)

    spher_all[:,9]=np.log(spher_all[:,9]+1)


    return spher_all,np.min(spher_all[:,0]),np.min(spher_all[:,3+0]),np.min(spher_all[:,6+0])
    
    
def Undo_Preprocess_1(spher_all,min_0,min_3,min_6):


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




def BackToCartesian(LHCcoord):

    #Muons1
    
    pt_mu1=LHCcoord[:,0]
    rapid_mu1=LHCcoord[:,1]
    azim_mu1=LHCcoord[:,2]

    px_mu1=pt_mu1*np.cos(azim_mu1)
    py_mu1=pt_mu1*np.sin(azim_mu1)
    pz_mu1=pt_mu1*np.sinh(rapid_mu1)
    E_mu1=pt_mu1*np.cosh(rapid_mu1)
    
    coord_mu1=np.array([E_mu1,px_mu1,py_mu1,pz_mu1])
    print(coord_mu1)
    print(np.shape(coord_mu1))
    coord_mu1=np.transpose(coord_mu1)
    
    
    #Muons2
    

    pt_mu2=LHCcoord[:,3+0]
    rapid_mu2=LHCcoord[:,3+1]
    azim_mu2=LHCcoord[:,3+2]

    px_mu2=pt_mu2*np.cos(azim_mu2)
    py_mu2=pt_mu2*np.sin(azim_mu2)
    pz_mu2=pt_mu2*np.sinh(rapid_mu2)
    E_mu2=pt_mu2*np.cosh(rapid_mu2)
    
    coord_mu2=np.array([E_mu2,px_mu2,py_mu2,pz_mu2])
    print(coord_mu2)
    print(np.shape(coord_mu2))
    coord_mu2=np.transpose(coord_mu2)
    
    
    #jets1
    
    
    pt_jet1=LHCcoord[:,6+0]
    rapid_jet1=LHCcoord[:,6+1]
    azim_jet1=LHCcoord[:,6+2]
    m_jet1=LHCcoord[:,6+3]
    
    
    
    mt_jet1=np.sqrt(pt_jet1**2+m_jet1**2)
    
    px_jet1=pt_jet1*np.cos(azim_jet1)
    py_jet1=pt_jet1*np.sin(azim_jet1)
    pz_jet1=mt_jet1*np.sinh(rapid_jet1)
    E_jet1=mt_jet1*np.cosh(rapid_jet1)
    
    coord_jet1=np.array([E_jet1,px_jet1,py_jet1,pz_jet1])
    print(coord_jet1)
    print(np.shape(coord_jet1))
    coord_jet1=np.transpose(coord_jet1)
    
    coord_all=np.concatenate((coord_mu1,coord_mu2,coord_jet1),axis=1)
    
    
    
    indexList = [np.any(i) for i in np.isinf(coord_all)]
    coord_all = np.delete(coord_all, indexList, axis=0)
    
    indexList = [np.any(i) for i in np.isnan(coord_all)]
    orig_coord = np.delete(coord_all, indexList, axis=0)
    

    return orig_coord

def SaveCoord(spher_all):


    file = h5py.File('LHCCoordZjet1.h5', 'w')
    file.create_dataset('LHCCoordZjet1', data=spher_all)
    file.close()
    return

spher_all = CoordTransforms(events)
spher_all_orig=spher_all
SaveCoord(spher_all)
name='corner_plot_sphercoord.png'
cornerplotter(spher_all,name)


cart_coord=BackToCartesian(spher_all_orig)
name='cart_vs_undo.png'
cornerplotter_comp(events[:100000,:],cart_coord[:100000,:],name)



spher_all_prep,min_0,min_3,min_6=Preprocess_1(spher_all_orig)
name='corner_plot_sphercoord_preprocessed_nomin.png'
cornerplotter(spher_all[:1000000,:],name)


spher_all_undo=Undo_Preprocess_1(spher_all_prep,min_0,min_3,min_6)
name='corner_plot_sphercoord_postprocessed_nomin.png'
cornerplotter(spher_all_undo[:1000000,:],name)


name='orig_vs_undo_nomin.png'
cornerplotter_comp(spher_all_orig[:1000000,:],spher_all_undo[:1000000,:],name)

cart_coord_afterundo=BackToCartesian(spher_all_undo)
name='cart_vs_undo_afterundo_nomin.png'
cornerplotter_comp(events[:1000000,:],cart_coord[:1000000,:],name)
name='cart_vs_undo_afterundo_marginal_nomin.png'
marginal_plot(events[:100000,:],cart_coord_afterundo[:100000,:],name)

exit()




