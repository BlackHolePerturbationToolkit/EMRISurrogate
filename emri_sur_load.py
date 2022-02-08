"""
********************************************************
**	// EMRI Sur Waveform           //          *****
**	// Tousif Islam                // 	   *****
**	// Date: 1st November, 2019    //	   *****
********************************************************

This Part of the code loads the surrogate data 
i.e.
        the value of {h_eim_amp_spline, h_eim_ph_spline, eim_indicies_amp, eim_indicies_ph, B_amp, B_ph} 
        and {time_array} obtained from training data using 

codes based on notebook `main_ALL_modes_long_duration_bump_fix_before_only_EMRI.ipynb' written by Nur-E-Mohammad Rifat 
modes={21,22,31,32,33,42,43,44,53,54,55}
"""
#----------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.interpolate import splrep, splev
import h5py
import hashlib
from gwtools import gwtools as _gwtools
import os
from os import path
#----------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------
def md5(fname):
    """ Compute has from file. code taken from 
    https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file"""
    
    # download file if not already there
    if path.isfile('EMRISur1dq1e4.h5')==False:
        print('EMRISur1dq1e4.h5 file is not found in the directory')
        print('... downloading h5 file from zenodo')
        os.system('wget https://zenodo.org/record/3612600/files/EMRISur1dq1e4.h5')
    
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

#----------------------------------------------------------------------------------------------------
def load_surrogate(EMRI):
    """ Loads all interpolation data for the following modes

    modes={21,22,31,32,33,42,43,44,53,54,55}

    Assumes the file EMRISur1dq1e4.h5 is located in the same directory
    as this file."""

    if EMRI == True:

        file_hash = md5('EMRISur1dq1e4.h5')
        zenodo_current_hash = "d145958484738e0c7292e084a66a96fa"

        if file_hash != zenodo_current_hash:
            raise AttributeError("EMRISur1dq1e4.h5 out of date.\n Please download new version from https://zenodo.org/record/3592428")


        with h5py.File('EMRISur1dq1e4.h5', 'r') as f:
        
            modes=[(2,1),(2,2),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,3),(5,4),(5,5)]
        
            h_eim_amp_spline_dict = {}
            h_eim_ph_spline_dict = {}
            B_amp_dict = {}
            B_ph_dict = {}
            eim_indicies_ph_dict = {}
            eim_indicies_amp_dict = {}
            time=[]
        
            for mode in modes:
            
                lmode,mmode=mode
                
                eim_indicies_amp_dataset=f['l%s_m%s/eim_indicies'%(lmode,mmode)]
                eim_indicies_amp_dict[(mode)]=eim_indicies_amp_dataset[:]
                eim_indicies_ph_dataset=f['l%s_m%s/eim_indicies_phase'%(lmode,mmode)]
                eim_indicies_ph_dict[(mode)]=eim_indicies_ph_dataset[:]
                B_ph_dataset=f['l%s_m%s/B_phase'%(lmode,mmode)]
                B_ph_dict[(mode)]=np.transpose(B_ph_dataset[:])
                B_amp_dataset=f['l%s_m%s/B'%(lmode,mmode)]
                B_amp_dict[(mode)]=np.transpose(B_amp_dataset[:])
                time_dataset=f['l%s_m%s/times'%(lmode,mmode)]
                time=time_dataset[:]
                degree_dataset=f['l%s_m%s/degree'%(lmode,mmode)]
                degree=degree_dataset[:]
                knots_dataset=f['l%s_m%s/spline_knots'%(lmode,mmode)]
                knots=knots_dataset[:]
                h_spline_amp_dataset=f['l%s_m%s/fitparams_amp'%(lmode,mmode)]
                h_spline_amp=h_spline_amp_dataset[:]
                h_spline_ph_dataset=f['l%s_m%s/fitparams_phase'%(lmode,mmode)]
                h_spline_ph=h_spline_ph_dataset[:]

                h_eim_amp_spline_dict[(mode)]=[(knots, h_spline_amp[flag,:],int(degree)) for flag in range(len(eim_indicies_amp_dict[(mode)]))]
                h_eim_ph_spline_dict[(mode)]=[(knots, h_spline_ph[flag,:],int(degree)) for flag in range(len(eim_indicies_ph_dict[(mode)]))]
        
        
        return time, eim_indicies_amp_dict, eim_indicies_ph_dict, B_amp_dict, B_ph_dict, h_eim_amp_spline_dict, h_eim_ph_spline_dict

    if EMRI == False:
        return 0
    
    
#----------------------------------------------------------------------------------------------------
def amp_ph_to_comp(amp,phase):
    """ Takes the amplitude and phase of the waveform and
    computes the compose them together"""
    
    full_wf = amp*np.exp(1j*phase)
    return full_wf


#----------------------------------------------------------------------------------------------------
def alpha_scaling_h(q,h):
    """ Implements alpha-scaling to match NR """

    nu=q/(1.+q)**2
    alpha=1.0-1.352854*nu-1.223006*(nu**2)+8.601968*(nu**3)-46.74562*(nu**4)
    h_scaled=np.array(h)*alpha
    return h_scaled


#----------------------------------------------------------------------------------------------------
def alpha_scaling_time(q, time):
    """ Implements alpha-scaling to match NR """

    nu=q/(1.+q)**2
    alpha=1.0-1.352854*nu-1.223006*(nu**2)+8.601968*(nu**3)-46.74562*(nu**4)
    t_scaled=np.array(time)*alpha
    return t_scaled

#----------------------------------------------------------------------------------------------------
def slog_surrogate(q, h_eim_amp_spline, h_eim_ph_spline, eim_indicies_amp, eim_indicies_ph, B_amp, B_ph, calibrated):
    """ Compute the interpolated waveform for a single mode """
    
    h_eim_amp = np.array([splev(np.log(q), h_eim_amp_spline[j])  for j in range(len(eim_indicies_amp))])
    h_eim_ph = np.array([splev(np.log(q), h_eim_ph_spline[j]) for j in range(len(eim_indicies_ph))])
    h_approx_amp = np.dot(B_amp.transpose(), h_eim_amp)
    h_approx_ph = np.dot(B_ph.transpose(), h_eim_ph)
    h_approx = amp_ph_to_comp(h_approx_amp, h_approx_ph)
    
    if calibrated==True:
        h_approx = alpha_scaling_h(q,h_approx) 
        
    return np.array(h_approx)*(1/q) # because the training waveform follows definition q<1 and we follow q>1


#----------------------------------------------------------------------------------------------------
def surrogate(modes, q_input, eim_indicies_amp_dict, eim_indicies_ph_dict, B_amp_dict, B_ph_dict, h_eim_amp_spline_dict, h_eim_ph_spline_dict, calibrated):
    """ Takes the interpolation indices, spline nodes, matrix B and computes the interpolated waveform for all modes"""
    
    h_approx={}
    for mode in modes:
        h_approx[(mode)] = slog_surrogate(q_input, h_eim_amp_spline_dict[(mode)], h_eim_ph_spline_dict[(mode)], eim_indicies_amp_dict[(mode)], eim_indicies_ph_dict[(mode)], B_amp_dict[(mode)], B_ph_dict[(mode)], calibrated)
        h_approx[(mode)] = np.array(np.conj(h_approx[(mode)])) # needed to match convention of other surrogate models
        
    return h_approx


#---------------------------------------------------------------------------------------------------- 
def geo_to_SI(t_geo, h_geo, M_tot, dist_mpc):
    """
    transforms the waveform from geomeric unit to physical unit
    given geoemtric time, geometric waveform, total mass M, distance dL
    """    
    # Physical units
    G=_gwtools.G
    MSUN_SI = _gwtools.MSUN_SI
    PC_SI = _gwtools.PC_SI
    C_SI = _gwtools.c
    M = M_tot * MSUN_SI
    dL = dist_mpc * PC_SI
    
    # scaling of time and h(t)
    t_SI = t_geo * (G*M/C_SI**3)
    
    strain_geo_to_SI = (G*M/C_SI**3)/dL
    h_SI={}
    for mode in h_geo.keys():
        h_SI[(mode)] = np.array(h_geo[mode])*strain_geo_to_SI
    
    return t_SI, h_SI
    


#---------------------------------------------------------------------------------------------------- 
def generate_surrogate(q_input, modes=[(2,1),(2,2),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,3),(5,4),(5,5)], \
                                 M_tot=None, dist_mpc=None, calibrated=True):
    """ 
    Description : Top-level function to generate surrogate waveform in either geometric or physical units
    
    Inputs
    ====================
    q_input : mass ratio
    
    modes : list of modes
            default is all available modes in the model i.e. [(2,1),(2,2),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,3),(5,4),(5,5)]
            
    M_total : total mass of the binary in solar unit
              default: None (in which case geometric wf is returned)
    
    dist_mpc : distance of the binary from the observer in Mpc
               default: None (in which case geometric wf is returned)
    
    calibrated : tell whether you want NR calibrated waveform or not
                 When set to True, it applies a scaling to the raw surrogate waveform 
                 This scaling has been obtained by calibrating the ppBHPT waveforms to NR in comparable mass ratio regime (1<=q<=10)
                 If set to False, the raw (uncalibrated)  ppBHPT waveforms are returned.
                 default: True
                 
    Output
    ====================
    t : time
    h : waveform modes
                 
    Example Uses:
    ====================
    1. to obtain NR Calibrated geometric waveform
            t, h = generate_surrogate(q_input, modes=[(2,1),(2,2),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,3),(5,4),(5,5)])
    2. to obtain raw geometric waveform
            t, h = generate_surrogate(q_input, modes=[(2,1),(2,2),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,3),(5,4),(5,5)], calibrated=False)       
    3. to obtain NR calibrated physical waveform
            t, h = generate_surrogate(q_input, modes=[(2,1),(2,2),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,3),(5,4),(5,5)], M_tot=50, dist_mpc=100)
    4. to obtain raw physical waveform
            t, h = generate_surrogate(q_input, modes=[(2,1),(2,2),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,3),(5,4),(5,5)], M_tot=50, dist_mpc=100, calibrated=False)
            
    """
    
    # geometric waveforms
    h_approx = surrogate(modes, q_input, eim_indicies_amp_dict, eim_indicies_ph_dict, B_amp_dict, B_ph_dict, h_eim_amp_spline_dict, h_eim_ph_spline_dict, calibrated)
    if calibrated==True:
        t_approx=alpha_scaling_time(q_input, time)
    else:
        t_approx=np.array(time)
    
    # relevant for obtaining physical waveforms
    if M_tot is not None and dist_mpc is not None:
        t_approx, h_approx = geo_to_SI(t_approx, h_approx, M_tot, dist_mpc)
    
    # add checks    
    elif M_tot is not None and dist_mpc is None:
        raise ValueError("Both M_tot and dist_mpc should be None! Or both should have physical values to generate physical waveform")
    elif M_tot is None and dist_mpc is not None:
        raise ValueError("Both M_tot and dist_mpc should be None! Or both should have physical values to generate physical waveform")
    
    return t_approx, h_approx


#----------------------------------------------------------------------------------------------------
# Calls the load surrogate function once called the data is loaded
time, eim_indicies_amp_dict, eim_indicies_ph_dict, B_amp_dict, B_ph_dict, h_eim_amp_spline_dict, h_eim_ph_spline_dict = load_surrogate(EMRI=True) 