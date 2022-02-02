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
#----------------------------------------------------------------------------------------------------


def md5(fname):
    """ Compute has from file. code taken from 
    https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file"""
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


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
        
            h_eim_amp_spline   = {}
            h_eim_ph_spline    = {}
            B_amp = {}
            B_ph = {}
            eim_indicies_ph = {}
            eim_indicies_amp = {}
            time=[]
        
            for mode in modes:
            
                lmode,mmode=mode
                
                eim_indicies_amp_dataset=f['l%s_m%s/eim_indicies'%(lmode,mmode)]
                eim_indicies_amp[(mode)]=eim_indicies_amp_dataset[:]
                eim_indicies_ph_dataset=f['l%s_m%s/eim_indicies_phase'%(lmode,mmode)]
                eim_indicies_ph[(mode)]=eim_indicies_ph_dataset[:]
                B_ph_dataset=f['l%s_m%s/B_phase'%(lmode,mmode)]
                B_ph[(mode)]=np.transpose(B_ph_dataset[:])
                B_amp_dataset=f['l%s_m%s/B'%(lmode,mmode)]
                B_amp[(mode)]=np.transpose(B_amp_dataset[:])
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

                h_eim_amp_spline[(mode)]=[(knots, h_spline_amp[flag,:],int(degree)) for flag in range(len(eim_indicies_amp[(mode)]))]
                h_eim_ph_spline[(mode)]=[(knots, h_spline_ph[flag,:],int(degree)) for flag in range(len(eim_indicies_ph[(mode)]))]
        
        return time, eim_indicies_amp, eim_indicies_ph, B_amp, B_ph, h_eim_amp_spline, h_eim_ph_spline

    if EMRI == False:
        return 0

def amp_ph_to_comp(a,ph):
    """ Takes the amplitude and phase of the waveform and
    computes the compose them together"""
#     import cmath as c
#     t =[]
#     for i in range(len(a)):
#         t.append(a[i]*c.exp(ph[i]*1j))
    t = a*np.exp(1j*ph)
    return t

def alpha_scaling_h(q,h):
    """ Implements alpha-scaling to match NR """

    nu=q/(1.+q)**2
    alpha=1.0-1.352854*nu-1.223006*nu*nu+8.601968*nu*nu*nu-46.74562*nu*nu*nu*nu
    h_scaled=np.array(h)*alpha
    return h_scaled

def alpha_scaling_time(q):
    """ Implements alpha-scaling to match NR """

    nu=q/(1.+q)**2
    alpha=1.0-1.352854*nu-1.223006*nu*nu+8.601968*nu*nu*nu-46.74562*nu*nu*nu*nu
    t_scaled=np.array(time)*alpha
    return t_scaled

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

def surrogate(modes, q_input, h_eim_amp_spline, h_eim_ph_spline, eim_indicies_amp, eim_indicies_ph, B_amp, B_ph, calibrated):
    """ Takes the interpolation indices, spline nodes, matrix B and computes the interpolated waveform for all modes"""
    h_approx={}
    for mode in modes:
        h_approx[(mode)] = slog_surrogate(q_input, h_eim_amp_spline[(mode)], h_eim_ph_spline[(mode)], eim_indicies_amp[(mode)], eim_indicies_ph[(mode)], B_amp[(mode)], B_ph[(mode)], calibrated)
        h_approx[(mode)] = np.array(np.conj(h_approx[(mode)])) # needed to match convention of other surrogate models
        
    return h_approx

def generate_surrogate(q_input,modes=[(2,1),(2,2),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,3),(5,4),(5,5)], calibrated=True):
    """ Top-level function to evaluate surrogate waveform.
    When calibrated = True, a scaling parameter is used to calibrate the ppBHPT waveforms to NR in comparable mass ratio regime.
    calibrated = False, the raw(uncalibrated)  ppBHPT waveforms are returned."""
    
    h_approx = surrogate(modes,q_input, h_eim_amp_spline, h_eim_ph_spline, eim_indicies_amp, eim_indicies_ph, B_amp, B_ph, calibrated)
    if calibrated==True:
        time_approx=alpha_scaling_time(q_input)
        return time_approx, h_approx
    else:
        return np.array(time), h_approx

def generate_surrogate_physical(q_input,M_total=80,dis=100,modes=[(2,1),(2,2),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,3),(5,4),(5,5)], \
                                calibrated=True):
    """ Top-level function to evaluate surrogate waveform in physical units for a source of total mass (in solar masses) M_total at a given
    distance (parsec). When calibrated = True, returns a calibrated waveform to NR; calibrated = False, the raw ppBHPT waveforms are
    returned"""
    
    time_approx, h_approx = generate_surrogate(q_input, modes, calibrated)
    
    # Physical units
    G=_gwtools.G
    MSUN_SI = _gwtools.MSUN_SI
    PC_SI = _gwtools.PC_SI
    C_SI = _gwtools.c

    M=M_total*MSUN_SI
    dL=dis* PC_SI
    
    # scaling of time and h(t)
    time=time_approx*(G*M/C_SI**3)
    
    for mode in modes:
        h_approx[(mode)] = np.array(h_approx[mode])*(G*M/C_SI**3)/dL
    
    return time, h_approx

# Calls the load surrogate function once called the data is loaded
time,eim_indicies_amp, eim_indicies_ph, B_amp, B_ph, h_eim_amp_spline, h_eim_ph_spline=load_surrogate(EMRI=True)
