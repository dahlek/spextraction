#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


mre_file = '/path/jupiter.mre' # Change 'path'
# option to save pdf of plot at bottom

bar_per_atm = 1.01325

def chi_sq_finder(data,model):
    '''
    Takes two arrays of the same dimensions, calcualted the chi squared value
    '''
    return np.sum((data-model)**2/model)

def reduced_chi_sq_finder(data,data_error,model):
    '''
    Takes two arrays of the same dimensions, calcualted the chi squared value
    Modeled after chi2.pro by James Sinclair
    '''
    N = len(data) # number of data points
    tmp = (data-model)/data_error
    chi_sq = np.sum(tmp**2)/N
    return chi_sq

def reduced_chi_sq_finder_2(data,model,num_fitted_param):
    '''
    Takes two arrays of the same dimensions, calcualted the chi squared value
    Uses the normal chi squared and the number of degrees of freedom (# wavelengths - # of fitted parameters)
    fitted parameters = 7 for CB only, 19 including chromo var
    '''
    N = len(data) # number of data points
    deg_of_freedom = N-num_fitted_param
    chisq = chi_sq_finder(data,model)
    chi_sq_red = chisq/deg_of_freedom
    return chi_sq_red


with open(mre_file,'r') as f:
    all_data=[x.split() for x in f.readlines()]
    #print all_data[0] # print the first line. will be a list.
    nlines=int(all_data[1][2]) # length of spectrum
    n_spec = int(all_data[1][1])
    spec=np.asfarray(all_data[5:5+nlines]) # Pull all the spectral data, the bulk of the mre file !!!Hard coded distance from the top of the file
    num_spectral_points = int(nlines/n_spec)

    R_meas_complete = spec[:,2]
    error_complete = spec[:,3]
    R_fit_complete = spec[:,5]
    chisq = reduced_chi_sq_finder(R_meas_complete,error_complete,R_fit_complete)
    chisq_list = chisq

print('Total reduced chisq = ',chisq_list)


colors = plt.cm.plasma(np.linspace(0,1,n_spec+2)) # define colors of measured spectra; replace "plasma" with colormap of choice


with open(mre_file,'r') as f:
    all_data=[x.split() for x in f.readlines()]
    #print all_data[0] # print the first line. will be a list.
    nlines=int(all_data[1][2]) # length of spectrum
    n_spec = int(all_data[1][1])
    spec=np.asfarray(all_data[5:5+nlines]) # Pull all the spectral data, the bulk of the mre file !!!Hard coded distance from the top of the file
    num_spectral_points = int(nlines/n_spec)

    x_index = 0    
    y_index = 0
    # Define complete spectra for chi squared calculation
    wl_complete = spec[:,1]
    R_meas_complete = spec[:,2]
    error_complete = spec[:,3]
    R_fit_complete = spec[:,5]
    
    # set up subplots; assuming 2
    if n_spec == 2:
        fig, axs = plt.subplots(2,n_spec,figsize=(10,6),dpi=200)
        
for i in range(0,n_spec):
    wl = spec[i*num_spectral_points:(i*num_spectral_points+num_spectral_points),1]  # wavelength in microns
    R_meas = spec[i*num_spectral_points:(i*num_spectral_points+num_spectral_points),2] # Measured radiance (data)
    error = spec[i*num_spectral_points:(i*num_spectral_points+num_spectral_points),3] # error
    R_fit = spec[i*num_spectral_points:(i*num_spectral_points+num_spectral_points),5] # Fit radiance

    if i%2 == 0: # even - plot in first column
        j = 0 # first column
        axs[0,j].plot(wl,R_meas,color=colors[0], label='Data',linewidth=2,marker='o')
        axs[0,j].plot(wl,R_fit,color=colors[1], label='Fit',linewidth=2,marker='o') 
        axs[0,j].fill_between(wl,R_meas+error,R_meas-error,color='lightgray')
        axs[0,j].legend(title='$\mu$=0.5-0.625')
        # plot residuals
        axs[1,j].plot(wl,R_meas-R_fit,color=colors[j+1],label='Residual',marker='o')
        axs[1,j].plot(wl,np.linspace(0,0,len(wl)),color='gray',linestyle='--',linewidth=2)
        axs[1,j].fill_between(wl,0+error,0-error,color='lightgray')
        axs[1,j].legend()
        
        
    if i%2 != 0: # even - plot in first column
        j = 1 # second column
        axs[0,j].plot(wl,R_meas,color=colors[0], label='Data',linewidth=2,marker='o')
        axs[0,j].plot(wl,R_fit,color=colors[1], label='Fit',linewidth=2,marker='o') 
        axs[0,j].fill_between(wl,R_meas+error,R_meas-error,color='lightgray')
        axs[0,j].legend(title='$\mu$=0.875-1.0')
        # plot residuals
        axs[1,j].plot(wl,R_meas-R_fit,color=colors[j+1],label='Residual',marker='o')
        axs[1,j].plot(wl,np.linspace(0,0,len(wl)),color='gray',linestyle='--',linewidth=2)
        axs[1,j].fill_between(wl,0+error,0-error,color='lightgray')
        axs[1,j].legend()
        
        
axs[1,0].set_ylabel('Residual')
axs[0,0].set_ylabel('I/F')

axs[1,0].set_xlabel('Wavelength (microns)')
axs[1,1].set_xlabel('Wavelength (microns)')

plt.tight_layout()
plt.subplots_adjust(hspace=0.0)

# plt.savefig('./2019jun1_fits.pdf',format='pdf')

