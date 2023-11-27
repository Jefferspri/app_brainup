import math
import pandas as pd
from datetime import datetime
import numpy as np
from collections import deque

from scipy import signal
from scipy.signal import filtfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import spkit as sp
from scipy.stats import skew
from scipy.fft import fft, fftfreq
from python_speech_features import mfcc

import pywt
from scipy import signal
import statistics as st

import os
from natsort import os_sorted




# generate date range and RT mean  
def generate_df_rt_date(df_rt):
    import math
    df_rt = df_rt.copy()
    cont = 0
    lst_tr_mean = []
    lst_f_tr_mean = []
    lst_date_start = []
    lst_date_end = []
    for i in range(df_rt.shape[0]):

        if df_rt['tag'].iloc[i] != 'click':
            cont +=1
            if cont == 1 :
                lst_date_start.append(df_rt['time'].iloc[i])
        
        if not math.isnan(df_rt['tr'].iloc[i]):
            
            rt = df_rt['tr'].iloc[i]
            lst_tr_mean.append(rt)

        if cont == 10: # 10
            lst_date_end.append(df_rt['time'].iloc[i])
            lst_f_tr_mean.append(np.mean(lst_tr_mean))
            lst_tr_mean = []
            cont = 0

    if cont != 0:
        lst_date_start = lst_date_start[:-1]

    df_rt_date = pd.DataFrame()
    df_rt_date['start'] = lst_date_start
    df_rt_date['end'] = lst_date_end
    df_rt_date['rt'] = lst_f_tr_mean
    
    df_rt_date = df_rt_date[(df_rt_date['rt']>=0.56)&(df_rt_date['rt']<=1.12)] # filter only valid answers
    df_rt_date.reset_index(inplace=True, drop=True)
    
    return df_rt_date



# another option of generate_df_rt_date: in this case we generate date range and the RT of each probe
def generate_df_rt_date_no_mean(df_rt):
    df_rt = df_rt.copy()
    dic_details = {'start':[], 'end':[], 'rt':[], 'flag':[]}
    
    if df_rt['tag'].iloc[-1] == 'click':
        df_rt = df_rt.iloc[:-1,:]
    
    for i in range(df_rt.shape[0]-1):
        if df_rt['tag'].iloc[i] != 'click':
            dic_details['start'].append(df_rt['time'].iloc[i])
            
            if df_rt['tag'].iloc[i+1] == 'click':
                dic_details['end'].append(df_rt['time'].iloc[i+2])
                dic_details['flag'].append(df_rt['flag'].iloc[i+2])
                
                if not math.isnan(df_rt['tr'].iloc[i+2]):
                    rt = df_rt['tr'].iloc[i+2]
                    dic_details['rt'].append(rt)
                else:
                    dic_details['rt'].append(float('nan')) # no rt availabe: only avilable for correct comission
                    
            else:
                dic_details['end'].append(df_rt['time'].iloc[i+1])
                dic_details['flag'].append(df_rt['flag'].iloc[i+1])
                
                if not math.isnan(df_rt['tr'].iloc[i+1]):
                    rt = df_rt['tr'].iloc[i+1]
                    dic_details['rt'].append(rt)
                else:
                    dic_details['rt'].append(float('nan')) # no rt availabe: only avilable for correct comission
                    
        
    df_rt_date = pd.DataFrame(dic_details)
    mask = []
    
    # filter only valid answers
    for n in range(df_rt_date.shape[0]):
        if not math.isnan(df_rt_date['rt'].iloc[n]):
            if (df_rt_date['rt'].iloc[n]>=0.45)&(df_rt_date['rt'].iloc[n]<=1.12): 
                mask.append(True)
            else:
                mask.append(False)
        else:
            mask.append(True)
            
    df_rt_date = df_rt_date[mask] 
    df_rt_date.reset_index(inplace=True, drop=True)

    return df_rt_date


# Interpolate missing reaction times using the average of proximal values.
def interp_rt(df_rt_date):
    df_rt = df_rt_date.copy()
    df_rt['rt'] = df_rt['rt'].fillna(0)
    RT_array = df_rt['rt'].to_numpy()
 
    for idx, val in enumerate(RT_array):
        if val == 0:
            idx_next_val = 1
            try:
                while RT_array[idx + idx_next_val] == 0:  # Find next non-zero value
                    idx_next_val += 1
                if idx == 0:  # If first value is zero, use the next non-zero value
                    RT_array[idx] = RT_array[idx + idx_next_val]
                else:  # else use the average of the two nearest non-zero
                    RT_array[idx] = (RT_array[idx - 1] + RT_array[idx + idx_next_val])/2
            except IndexError:  # If end of file is reached, use the last non-zero
                RT_array[idx] = RT_array[idx - 1]
    
    df_rt['rt'] = RT_array
    # Return non nan value.
    return df_rt

def generate_rt_mean(df_rt_date):
    # Calculate the mean of each group of 10 numbers and replace the 'vtc' column
    group_size = 10
    num_groups = df_rt_date.shape[0]//group_size
    new_range = {'start':[],'end':[],'vtc':[]}
    for i in range(num_groups):
        group_start = i * group_size
        group_end = (i + 1) * group_size
        group_mean = df_rt_date['vtc'][group_start:group_end].mean()
        new_range['start'].append(df_rt_date['start'].iloc[group_start])
        new_range['end'].append(df_rt_date['end'].iloc[group_end-1])
        new_range['vtc'].append(group_mean)
    
    return pd.DataFrame(new_range)

# Compute the variance time course (VTC) of the array RT_interp
def compute_VTC(df_rt_date):
    df_rt = df_rt_date.copy()
    VTC = abs(df_rt['rt'] - df_rt['rt'].mean(skipna=True))/df_rt['rt'].std(skipna=True)
    #VTC = VTC.fillna(0)
    #VTC = VTC.interpolate()

    def fwhm2sigma(fwhm=9):
        return fwhm / np.sqrt(8 * np.log(2))
    
    filt = signal.gaussian(len(VTC), fwhm2sigma(9))
    VTC_filtered = np.convolve(VTC, filt, "same") / sum(filt)
    
    df_rt['vtc_noise'] = VTC
    df_rt['vtc'] = VTC_filtered

    return df_rt 

# Compute the variance time course (VTC) of the array RT_interp
def compute_VTC_validity(df_rt_date):
    df_rt = df_rt_date.copy()
    VTC = abs(df_rt['rt'] - df_rt['rt'].mean(skipna=True))/df_rt['rt'].std(skipna=True)

    def fwhm2sigma(fwhm=9):
        return fwhm / np.sqrt(8 * np.log(2))
    
    filt = signal.gaussian(len(VTC), fwhm2sigma(9))
    VTC_filtered = np.convolve(VTC, filt, "same") / sum(filt)

    df_rt['vtc'] = VTC_filtered

    return df_rt 


def calculate_mean_vtc(df_rt_date):
    # Calculate the mean of each group of 10 numbers and replace the 'vtc' column
    group_size = 10
    num_groups = df_rt_date.shape[0]//group_size
    new_range = {'start':[],'end':[],'vtc':[]}
    for i in range(num_groups):
        group_start = i * group_size
        group_end = (i + 1) * group_size
        group_mean = df_rt_date['vtc'][group_start:group_end].mean()
        new_range['start'].append(df_rt_date['start'].iloc[group_start])
        new_range['end'].append(df_rt_date['end'].iloc[group_end-1])
        new_range['vtc'].append(group_mean)
    
    return pd.DataFrame(new_range)


# Preprocessing EEG data
def preprocessimg_data(df_eeg):
    
    df_eeg = df_eeg.copy()
    
    # Applying notch filter in 60Hz.
    b_notch = [0.9879, -0.1937, 0.9879]
    a_notch = [1.0, -0.1937, 0.9758]
    size = df_eeg['TP9'].shape[0]
  
    df_eeg['TP9_fil'] = filtfilt(b_notch, a_notch, df_eeg['TP9'], padlen=size-1)
    df_eeg['TP10_fil'] = filtfilt(b_notch, a_notch, df_eeg['TP10'], padlen=size-1)
    df_eeg['AF7_fil'] = filtfilt(b_notch, a_notch, df_eeg['AF7'], padlen=size-1)
    df_eeg['AF8_fil'] = filtfilt(b_notch, a_notch, df_eeg['AF8'], padlen=size-1)

    # Applying high pass filter in 1Hz.
    b_high, a_high = signal.butter(5, 1, 'hp', fs=256)
    df_eeg['TP9_fil'] = filtfilt(b_high, a_high, df_eeg['TP9_fil'], padlen=size-1)
    df_eeg['TP10_fil'] = filtfilt(b_high, a_high, df_eeg['TP10_fil'], padlen=size-1)
    df_eeg['AF7_fil'] = filtfilt(b_high, a_high, df_eeg['AF7_fil'], padlen=size-1)
    df_eeg['AF8_fil'] = filtfilt(b_high, a_high, df_eeg['AF8_fil'], padlen=size-1)
    
    # Filter eye noise
    df_eeg['TP9_fil'] = sp.eeg.ATAR(df_eeg['TP9_fil'], wv='db9',beta=0.1,OptMode='elim',verbose=1,k1=10, k2=100, winsize=256)
    df_eeg['TP10_fil'] = sp.eeg.ATAR(df_eeg['TP10_fil'], wv='db9',beta=0.1,OptMode='elim',verbose=1,k1=10, k2=100, winsize=256)
    df_eeg['AF7_fil'] = sp.eeg.ATAR(df_eeg['AF7_fil'], wv='db9',beta=0.1,OptMode='elim',verbose=1,k1=10, k2=100, winsize=256)
    df_eeg['AF8_fil'] = sp.eeg.ATAR(df_eeg['AF8_fil'], wv='db9',beta=0.1,OptMode='elim',verbose=1,k1=10, k2=100, winsize=256)

    return df_eeg


# Calculate Zero Crossing Rate 
def calculate_zcr(signal, type_z):
    if type_z == 'zcr':
        return np.count_nonzero(np.diff(np.sign(signal))==0)/len(signal)
    elif type_z == 'direction':
        if len(signal) >= 11:
            return (np.where(np.diff(np.sign(np.diff(list(signal)[1:]))) != 0)[0].size)/10  # rolling direction change of ZCR
        else:
            return np.nan

def compute_energy_entropy(window, num_short_blocks):
      """
      Computes the energy entropy of the given frame.

      Args:
        window: A NumPy array containing the audio samples of the input frame.
        num_short_blocks: The number of sub-frames.

      Returns:
        A NumPy float containing the energy entropy value.
      """

      # Compute the total frame energy.
      total_energy = np.sum(window**2)

      # Compute the window length.
      window_length = len(window)

      # Compute the sub-window length.
      sub_window_length = window_length // num_short_blocks

      # If the window length is not a perfect multiple of the sub-window length,
      # truncate the window to the nearest multiple.
      if window_length != sub_window_length*num_short_blocks :
            window = window[:sub_window_length * num_short_blocks]

      # Reshape the window into sub-windows.
      sub_windows = window.reshape(sub_window_length, num_short_blocks)

      # Compute the normalized sub-frame energies.
      normalized_sub_frame_energies = np.sum(sub_windows**2, axis=0) / (total_energy + np.finfo(float).eps)

      # Compute the entropy of the normalized sub-frame energies.
      entropy = -np.sum(normalized_sub_frame_energies * np.log2(normalized_sub_frame_energies + np.finfo(float).eps))

      return entropy


def compute_spectral_entropy(window, num_short_blocks):
      """
      Computes the energy entropy of the given frame.

      Args:
        window: A NumPy array containing the audio samples of the input frame.
        num_short_blocks: The number of sub-frames.

      Returns:
        A NumPy float containing the energy entropy value.
      """
      N = len(window)
      window = 2.0/N *np.abs(fft(window)[0:N//2])

      # Compute the total frame energy.
      total_energy = np.sum(window**2)

      # Compute the window length.
      window_length = len(window)

      # Compute the sub-window length.
      sub_window_length = window_length // num_short_blocks
      #print(window_length, sub_window_length)

      # If the window length is not a perfect multiple of the sub-window length,
      # truncate the window to the nearest multiple.
      if window_length != sub_window_length*num_short_blocks :
            window = window[:sub_window_length * num_short_blocks]

      # Reshape the window into sub-windows.
      sub_windows = window.reshape(sub_window_length, num_short_blocks)

      # Compute the normalized sub-frame energies.
      normalized_sub_frame_energies = np.sum(sub_windows**2, axis=0) / (total_energy + np.finfo(float).eps)

      # Compute the entropy of the normalized sub-frame energies.
      entropy = -np.sum(normalized_sub_frame_energies * np.log2(normalized_sub_frame_energies + np.finfo(float).eps))

      return entropy, total_energy


def compute_spectral_rolloff(window, c):
    """Computes the spectral rolloff feature.

    Args:
        window_fft: A NumPy array containing the magnitude spectrum of the input window.
        c: The spectral rolloff parameter.

    Returns:
        A NumPy float containing the spectral rolloff value.
    """
    N = len(window)
    window_fft = 2.0/N *np.abs(fft(window)[0:N//2])
    xf = fftfreq(N, 1/256)[:N//2]

    # Compute the total spectral energy.
    total_energy = np.sum(window_fft**2)

    # Compute the spectral rolloff as the frequency position where the
    # cumulative spectral energy is equal to c * total_energy.
    cur_energy = 0.0
    count_fft = 0
    fft_len = len(window_fft)

    while cur_energy <= c*total_energy and count_fft <= fft_len:
        cur_energy = cur_energy  + window_fft[count_fft]**2
        count_fft += 1
        
    count_fft -= 1
    # Normalize the spectral rolloff to the length of the FFT.
    mC = count_fft #(count_fft - 0)/(128 - 0)  # normalize respect of 128, not necesary

    return mC

def mean_compute(lista):
    """
    Compute mean of last 10 values of determine value
    """
    if len(lista)==11:
        return np.mean(list(lista)[:10])
    else:
        return np.nan


# Pivot channels from rows to columns 
def pivot_channels(features):
    features =  features.copy()
    #features = features.dropna()

    df_all_features = pd.DataFrame()
    col_names = []

    # concatenate characteristics columns per channel
    for ch in ['TP9','TP10','AF7']: # 
        dfx = features[features['channel']== ch+'_fil' ]
        dfx.reset_index(inplace = True, drop = True)
        dfx = dfx.iloc[:,1:64]  # 26 + 7 + 11 = 44
        col_names = col_names + [ch+'_'+text for text in list(dfx.columns)]
        df_all_features = pd.concat([df_all_features, dfx], axis=1, ignore_index=True)
        
    dfx = features[features['channel']==  'AF8_fil']
    dfx.reset_index(inplace = True, drop = True)
    dfx = dfx.iloc[:,1:69] # 27 + 7 + 11 +2= 45
    col_names = col_names + ['AF8_'+text for text in list(dfx.columns)][:-5] + ['rt','vtc','class','time','flag']
    df_all_features = pd.concat([df_all_features, dfx], axis=1, ignore_index=True)

    df_all_features.columns = col_names
    #df_all_features = df_all_features.dropna()
    df_all_features.reset_index(drop=True, inplace=True)

    return df_all_features


# Wavelet descomposition and calculate characteristics 
def wavelet_packet_decomposition(df_eeg, df_rt_date):
    features = {'channel':[],'p_delta':[], 'p_theta':[],'p_alpha':[], 'p_beta':[], 'p_gamma':[],
                'p_max_delta':[], 'p_max_theta':[],'p_max_alpha':[], 'p_max_beta':[], 'p_max_gamma':[],
                'p_min_delta':[], 'p_min_theta':[],'p_min_alpha':[], 'p_min_beta':[], 'p_min_gamma':[],
                'p_beta_theta':[], 'p_beta_alpha':[], 'p_beta_alpha_theta':[],
                'std_delta':[],'std_theta':[], 'std_alpha':[], 'std_beta':[], 'std_gamma':[],
                'skew_delta':[],'skew_theta':[], 'skew_alpha':[], 'skew_beta':[], 'skew_gamma':[],
                'total_variation':[],'zcr':[], 'zcr_dir':[],
                'energy_entropy':[], 'spectral_entropy':[], 'total_energy':[], 'spectral_rollof':[], 'mfcc':[], 
                'ten_zcr':[], 'ten_p_delta':[], 'ten_p_theta':[], 'ten_p_alpha':[], 'ten_p_beta':[], 'ten_p_gamma':[], 'ten_total_variation':[], 'ten_energy_entropy':[], 'ten_spectral_entropy':[], 'ten_total_energy':[], 'ten_spectral_rollof':[], 'ten_mfcc':[], 
                'ten_pmax_delta':[], 'ten_pmax_theta':[], 'ten_pmax_alpha':[], 'ten_pmax_beta':[], 'ten_pmax_gamma':[],'ten_skew_delta':[], 'ten_skew_theta':[], 'ten_skew_alpha':[], 'ten_skew_beta':[], 'ten_skew_gamma':[],'ten_std_delta':[], 'ten_std_theta':[], 'ten_std_alpha':[], 'ten_std_beta':[], 'ten_std_gamma':[],
                'rt':[],'vtc':[],'class':[], 'time':[],'flag':[]}
    
    dic_slide ={'TP9_fil': {'zcr':deque(maxlen=11), 'p_delta':deque(maxlen=11), 'p_theta':deque(maxlen=11), 'p_alpha':deque(maxlen=11),'p_beta':deque(maxlen=11), 'p_gamma':deque(maxlen=11), 'total_variation':deque(maxlen=11), 'energy_entropy':deque(maxlen=11), 'spectral_entropy':deque(maxlen=11), 'total_energy':deque(maxlen=11), 'spectral_rollof':deque(maxlen=11), 'mfcc':deque(maxlen=11), 'pmax_delta':deque(maxlen=11), 'pmax_theta':deque(maxlen=11), 'pmax_alpha':deque(maxlen=11),'pmax_beta':deque(maxlen=11), 'pmax_gamma':deque(maxlen=11), 'skew_delta':deque(maxlen=11), 'skew_theta':deque(maxlen=11), 'skew_alpha':deque(maxlen=11),'skew_beta':deque(maxlen=11), 'skew_gamma':deque(maxlen=11), 'std_delta':deque(maxlen=11), 'std_theta':deque(maxlen=11), 'std_alpha':deque(maxlen=11),'std_beta':deque(maxlen=11), 'std_gamma':deque(maxlen=11)},
                'TP10_fil': {'zcr':deque(maxlen=11), 'p_delta':deque(maxlen=11), 'p_theta':deque(maxlen=11), 'p_alpha':deque(maxlen=11),'p_beta':deque(maxlen=11), 'p_gamma':deque(maxlen=11), 'total_variation':deque(maxlen=11), 'energy_entropy':deque(maxlen=11), 'spectral_entropy':deque(maxlen=11), 'total_energy':deque(maxlen=11), 'spectral_rollof':deque(maxlen=11), 'mfcc':deque(maxlen=11), 'pmax_delta':deque(maxlen=11), 'pmax_theta':deque(maxlen=11), 'pmax_alpha':deque(maxlen=11),'pmax_beta':deque(maxlen=11), 'pmax_gamma':deque(maxlen=11), 'skew_delta':deque(maxlen=11), 'skew_theta':deque(maxlen=11), 'skew_alpha':deque(maxlen=11),'skew_beta':deque(maxlen=11), 'skew_gamma':deque(maxlen=11), 'std_delta':deque(maxlen=11), 'std_theta':deque(maxlen=11), 'std_alpha':deque(maxlen=11),'std_beta':deque(maxlen=11), 'std_gamma':deque(maxlen=11)},
                'AF7_fil': {'zcr':deque(maxlen=11), 'p_delta':deque(maxlen=11), 'p_theta':deque(maxlen=11), 'p_alpha':deque(maxlen=11),'p_beta':deque(maxlen=11), 'p_gamma':deque(maxlen=11), 'total_variation':deque(maxlen=11), 'energy_entropy':deque(maxlen=11), 'spectral_entropy':deque(maxlen=11), 'total_energy':deque(maxlen=11), 'spectral_rollof':deque(maxlen=11), 'mfcc':deque(maxlen=11), 'pmax_delta':deque(maxlen=11), 'pmax_theta':deque(maxlen=11), 'pmax_alpha':deque(maxlen=11),'pmax_beta':deque(maxlen=11), 'pmax_gamma':deque(maxlen=11), 'skew_delta':deque(maxlen=11), 'skew_theta':deque(maxlen=11), 'skew_alpha':deque(maxlen=11),'skew_beta':deque(maxlen=11), 'skew_gamma':deque(maxlen=11), 'std_delta':deque(maxlen=11), 'std_theta':deque(maxlen=11), 'std_alpha':deque(maxlen=11),'std_beta':deque(maxlen=11), 'std_gamma':deque(maxlen=11)},
                'AF8_fil': {'zcr':deque(maxlen=11), 'p_delta':deque(maxlen=11), 'p_theta':deque(maxlen=11), 'p_alpha':deque(maxlen=11),'p_beta':deque(maxlen=11), 'p_gamma':deque(maxlen=11), 'total_variation':deque(maxlen=11), 'energy_entropy':deque(maxlen=11), 'spectral_entropy':deque(maxlen=11), 'total_energy':deque(maxlen=11), 'spectral_rollof':deque(maxlen=11), 'mfcc':deque(maxlen=11), 'pmax_delta':deque(maxlen=11), 'pmax_theta':deque(maxlen=11), 'pmax_alpha':deque(maxlen=11),'pmax_beta':deque(maxlen=11), 'pmax_gamma':deque(maxlen=11), 'skew_delta':deque(maxlen=11), 'skew_theta':deque(maxlen=11), 'skew_alpha':deque(maxlen=11),'skew_beta':deque(maxlen=11), 'skew_gamma':deque(maxlen=11), 'std_delta':deque(maxlen=11), 'std_theta':deque(maxlen=11), 'std_alpha':deque(maxlen=11),'std_beta':deque(maxlen=11), 'std_gamma':deque(maxlen=11)}}

    for i in range(df_rt_date.shape[0]):
        df_trans = df_eeg[(df_eeg['time']>=df_rt_date.iloc[i,0]) & (df_eeg['time']<df_rt_date.iloc[i,1])]
        if df_trans.shape[0]>=180: # only consider operate feature extracction if we have more than 190 EEG point for 0.8seg
            
            for channel in ['TP9_fil', 'TP10_fil', 'AF7_fil','AF8_fil']:
                chirp_signal = df_trans[channel].to_numpy()

                # Decomposing signal

                # [0, 64] [64, 128] 
                (B11, B12) = pywt.dwt(chirp_signal, 'db9', 'zero')

                # [0, 32] [32, 64]
                (B21, B22) = pywt.dwt(B11, 'db9', 'zero')
                # [64, 96] [96, 128]
                (B23, B24) = pywt.dwt(B12, 'db9', 'zero')

                # [0, 16] [16, 32]
                (B31, B32) = pywt.dwt(B21, 'db9', 'zero')

                # [0, 8] [8, 16]
                (B41, B42) = pywt.dwt(B31, 'db9', 'zero')

                # [0, 4] [4, 8]
                (B51, B52) = pywt.dwt(B41, 'db9', 'zero')
                # [8, 12] [12, 16]
                (B53, B54) = pywt.dwt(B42, 'db9', 'zero')

                # [12, 14] [14, 16]
                (B61, B62) = pywt.dwt(B54, 'db9', 'zero')

                # [12, 13] [13, 14]
                (B71, B72) = pywt.dwt(B61, 'db9', 'zero')

                # grouping signals
                group_delta = [B51] # [0, 4]
                group_theta = [B52] # [4, 8]
                group_alpha = [B71, np.zeros_like(B72), np.zeros_like(B62), B53] # [12, 13] [8, 12] 
                group_beta  = [B72, np.zeros_like(B71), B62, np.zeros_like(B54), np.zeros_like(B42), B32] # [13, 14] [14, 16] [16, 32]
                group_gamma = [B22, B23] # [32, 64] [64, 96]

                # reconstruction
                delta = pywt.waverec(group_delta, 'db9', 'zero') # [0, 4]
                theta = pywt.waverec(group_theta, 'db9', 'zero') # [4, 8]
                alpha = pywt.waverec(group_alpha, 'db9', 'zero') # [8, 13]
                beta = pywt.waverec(group_beta, 'db9', 'zero')   # [13, 32]
                gamma = pywt.waverec(group_gamma, 'db9', 'zero') # [32, 96]
                
                # ---------------------------------------- Compute Characteristics --------------------------------------- 
                # Welch’s power spectral density
                fs = 256
                (f, S_delta)= signal.welch(delta, fs, nperseg=len(delta)) # nperseg is the number of points that delta have
                (f, S_theta)= signal.welch(theta, fs, nperseg=len(theta))
                (f, S_alpha)= signal.welch(alpha, fs, nperseg=len(alpha))
                (f, S_beta)= signal.welch(beta, fs, nperseg=len(beta))
                (f, S_gamma)= signal.welch(gamma, fs, nperseg=len(gamma))

                # Classic features: β/θ, β/α, β/(α+θ) 
                beta_theta = sum(S_beta)/sum(S_theta)
                beta_alpha = sum(S_beta)/sum(S_alpha)
                beta_alpha_theta =  sum(S_beta)/(sum(S_alpha)+sum(S_theta))

                # Total Variation (TV)
                t_vari = np.sum(np.abs(np.diff(chirp_signal)))

                # Saving features
                     # Power per channel
                features['channel'].append(channel)
                features['p_delta'].append(sum(S_delta))
                features['p_theta'].append(sum(S_theta))
                features['p_alpha'].append(sum(S_alpha))
                features['p_beta'].append(sum(S_beta))
                features['p_gamma'].append(sum(S_gamma))
                dic_slide[channel]['p_delta'].append(features['p_delta'][-1])
                dic_slide[channel]['p_theta'].append(features['p_theta'][-1])
                dic_slide[channel]['p_alpha'].append(features['p_alpha'][-1])
                dic_slide[channel]['p_beta'].append(features['p_beta'][-1])
                dic_slide[channel]['p_gamma'].append(features['p_gamma'][-1])
                    # Max Power per channel
                features['p_max_delta'].append(max(S_delta))
                features['p_max_theta'].append(max(S_theta))
                features['p_max_alpha'].append(max(S_alpha))
                features['p_max_beta'].append(max(S_beta))
                features['p_max_gamma'].append(max(S_gamma))
                dic_slide[channel]['pmax_delta'].append(features['p_max_delta'][-1])
                dic_slide[channel]['pmax_theta'].append(features['p_max_theta'][-1])
                dic_slide[channel]['pmax_alpha'].append(features['p_max_alpha'][-1])
                dic_slide[channel]['pmax_beta'].append(features['p_max_beta'][-1])
                dic_slide[channel]['pmax_gamma'].append(features['p_max_gamma'][-1])
                    # Min Power per channel
                features['p_min_delta'].append(min(S_delta))
                features['p_min_theta'].append(min(S_theta))
                features['p_min_alpha'].append(min(S_alpha))
                features['p_min_beta'].append(min(S_beta))
                features['p_min_gamma'].append(min(S_gamma))
                    # Power ratios per channel
                features['p_beta_theta'].append(beta_theta)
                features['p_beta_alpha'].append(beta_alpha)
                features['p_beta_alpha_theta'].append(beta_alpha_theta)
                    # STD from temporal EEG
                features['std_delta'].append(delta.std())
                features['std_theta'].append(theta.std())
                features['std_alpha'].append(alpha.std())
                features['std_beta'].append(beta.std())
                features['std_gamma'].append(gamma.std())
                dic_slide[channel]['std_delta'].append(features['std_delta'][-1])
                dic_slide[channel]['std_theta'].append(features['std_theta'][-1])
                dic_slide[channel]['std_alpha'].append(features['std_alpha'][-1])
                dic_slide[channel]['std_beta'].append(features['std_beta'][-1])
                dic_slide[channel]['std_gamma'].append(features['std_gamma'][-1])
                    # Skeness
                features['skew_delta'].append(skew(delta))
                features['skew_theta'].append(skew(theta))
                features['skew_alpha'].append(skew(alpha))
                features['skew_beta'].append(skew(beta))
                features['skew_gamma'].append(skew(gamma))
                dic_slide[channel]['skew_delta'].append(features['skew_delta'][-1])
                dic_slide[channel]['skew_theta'].append(features['skew_theta'][-1])
                dic_slide[channel]['skew_alpha'].append(features['skew_alpha'][-1])
                dic_slide[channel]['skew_beta'].append(features['skew_beta'][-1])
                dic_slide[channel]['skew_gamma'].append(features['skew_gamma'][-1])
                    # Total Variation (TV)
                features['total_variation'].append(t_vari)
                dic_slide[channel]['total_variation'].append(t_vari)
                    # Zero-crossing rate
                zcr_mod = calculate_zcr(chirp_signal, type_z='zcr')
                features['zcr'].append(zcr_mod) 
                dic_slide[channel]['zcr'].append(zcr_mod)
                    # ZCR direction changes
                zcr_direction = calculate_zcr(dic_slide[channel]['zcr'], type_z='direction')
                features['zcr_dir'].append(zcr_direction)
                    # Energy Entropy
                features['energy_entropy'].append(compute_energy_entropy(window=chirp_signal, num_short_blocks=20))
                dic_slide[channel]['energy_entropy'].append(features['energy_entropy'][-1])
                    # Spectral Entropy
                entropy, total_energy = compute_spectral_entropy(window = chirp_signal, num_short_blocks = 10)
                features['spectral_entropy'].append(entropy)
                dic_slide[channel]['spectral_entropy'].append(entropy)
                    # Total Energy 
                features['total_energy'].append(total_energy)
                dic_slide[channel]['total_energy'].append(total_energy)
                    # Spectral Rollof
                features['spectral_rollof'].append(compute_spectral_rolloff(window = chirp_signal, c=0.5))
                dic_slide[channel]['spectral_rollof'].append(features['spectral_rollof'][-1])
                    # Mel 
                features['mfcc'].append(np.std(mfcc(chirp_signal,256, winlen=0.8, winstep=0.8, numcep=29)))
                dic_slide[channel]['mfcc'].append(features['mfcc'][-1])
                    # Nine means
                features['ten_zcr'].append(mean_compute(dic_slide[channel]['zcr']))
                features['ten_p_delta'].append(mean_compute(dic_slide[channel]['p_delta']))
                features['ten_p_theta'].append(mean_compute(dic_slide[channel]['p_theta']))
                features['ten_p_alpha'].append(mean_compute(dic_slide[channel]['p_alpha']))
                features['ten_p_beta'].append(mean_compute(dic_slide[channel]['p_beta']))
                features['ten_p_gamma'].append(mean_compute(dic_slide[channel]['p_gamma']))
                features['ten_total_variation'].append(mean_compute(dic_slide[channel]['total_variation']))
                features['ten_energy_entropy'].append(mean_compute(dic_slide[channel]['energy_entropy']))
                features['ten_spectral_entropy'].append(mean_compute(dic_slide[channel]['spectral_entropy']))
                features['ten_total_energy'].append(mean_compute(dic_slide[channel]['total_energy']))
                features['ten_spectral_rollof'].append(mean_compute(dic_slide[channel]['spectral_rollof']))
                features['ten_mfcc'].append(mean_compute(dic_slide[channel]['mfcc']))
                features['ten_pmax_delta'].append(mean_compute(dic_slide[channel]['pmax_delta']))
                features['ten_pmax_theta'].append(mean_compute(dic_slide[channel]['pmax_theta']))
                features['ten_pmax_alpha'].append(mean_compute(dic_slide[channel]['pmax_alpha']))
                features['ten_pmax_beta'].append(mean_compute(dic_slide[channel]['pmax_beta']))
                features['ten_pmax_gamma'].append(mean_compute(dic_slide[channel]['pmax_gamma']))
                features['ten_skew_delta'].append(mean_compute(dic_slide[channel]['skew_delta']))
                features['ten_skew_theta'].append(mean_compute(dic_slide[channel]['skew_theta']))
                features['ten_skew_alpha'].append(mean_compute(dic_slide[channel]['skew_alpha']))
                features['ten_skew_beta'].append(mean_compute(dic_slide[channel]['skew_beta']))
                features['ten_skew_gamma'].append(mean_compute(dic_slide[channel]['skew_gamma']))
                features['ten_std_delta'].append(mean_compute(dic_slide[channel]['std_delta']))
                features['ten_std_theta'].append(mean_compute(dic_slide[channel]['std_theta']))
                features['ten_std_alpha'].append(mean_compute(dic_slide[channel]['std_alpha']))
                features['ten_std_beta'].append(mean_compute(dic_slide[channel]['std_beta']))
                features['ten_std_gamma'].append(mean_compute(dic_slide[channel]['std_gamma']))

                    # Target Etiquetes
                features['rt'].append(df_rt_date['rt'].iloc[i]) 
                features['vtc'].append(df_rt_date['vtc'].iloc[i]) 
                features['class'].append(df_rt_date['class'].iloc[i])
                features['time'].append(df_rt_date['start'].iloc[i])
                features['flag'].append(df_rt_date['flag'].iloc[i])
    
    df_features = pd.DataFrame(features)
    #df_features = df_features.dropna()
    features = pivot_channels(df_features).bfill()

    return features

def z_normalization(df):
    df_zscore = (df - df.mean())/df.std()
    return df_zscore

def normalization_zero_to_one(df):
    df_score = (df - df.min())/(df.max() - df.min())
    return df_score

def normalization(df_features):
    for col in list(df_features.columns)[:-5]:
        df_features[col] = normalization_zero_to_one(df_features[col]) 

    return df_features


def plot_vtc(df_rt_date, ori_med):
    x = [i for i in range(df_rt_date.shape[0])]
    y = df_rt_date['vtc'].to_numpy()
    condition = y > ori_med  # Define your condition here

    fig, ax = plt.subplots()
    plt.plot(df_rt_date['vtc_noise'].to_list(),  color='#b5b5b5')
    # Initialize variables to track line segments
    segment_start = 0
    current_color = '#fc7303'  # Initial color: orange

    for i in range(1, len(x)):
        if condition[i] != condition[i - 1] or i == len(x) - 1:
            # When the condition changes or at the last data point
            segment_end = i
            segment_x = x[segment_start:segment_end + 1]
            segment_y = y[segment_start:segment_end + 1]

            # Plot the line segment with the current color
            ax.plot(segment_x, segment_y, color=current_color, linewidth=2)

            # Update the segment_start and current_color
            segment_start = i
            current_color = '#0012b0' if current_color == '#fc7303' else '#fc7303'  # Switch the color
    
    plt.axhline(ori_med, color='black')
    plt.xlabel('Sample [n]')
    plt.ylabel('Normalized RT Variability')
    plt.title('Line Plot with Two Colors based on Condition')
    #plt.grid(True)

    plt.show()