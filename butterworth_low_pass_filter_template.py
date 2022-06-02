#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------
# Created By  : Paul Aubin
# Created Date: 2022/05/29
# version ='1.0'
# header reference : https://www.delftstack.com/howto/python/low-pass-filter-python/
# ----------------------------------------------------------------------
""" Template of butterworth low-pass filter """
# ----------------------------------------------------------------------


import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def test_butter_manual(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    for i in range(0, data.size):
        if i >= order:
            #print('i = ', repr(i))
            #print('b = ', repr(b))
            #print('data[i-order:i]')
            #print('np.flip(data[i-order:i+1]) = ', repr(np.flip(data[i-order:i+1])))
            x_term = np.sum(b * np.flip(data[i-order:i+1]))
            y_term = np.sum(a[1:] * np.flip(y[i-order:i]))
            y[i] = 1/a[0] * (x_term - y_term)
        else:
            y[i] = 0
    return y

def test_butter_manual_2(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = manual_filter(b, a, data)
    print('y = ', repr(y))
    return y


def manual_filter(b, a, input, init=[]):
    """Manually computes a digital filter.

    Keyword arguments:
    b -- the numerator of a filter [double]
    a -- the denominator of a filter [double]
    input -- the data to be filtered [np.array]
    init -- the initial values of the filter [np.array]

    Output:
    y -- the filtered values of input
    """
    order = len(a)
    y = np.array(np.zeros(len(input)))
    if len(init) == 0:
        init = np.array(np.zeros(order))
    for i in range(0, len(input)):
        if i>= order - 1:
            x_term = np.sum(b * np.flip(input[i-order+1:i+1]))
            y_term = np.sum(a[1:] * np.flip(y[i-order+1:i]))
            y[i] = 1/a[0] * (x_term - y_term)
        else:
            y[i] = init[i]
    return y 
    

'''
# Setting standard filter requirements.
order = 2 #6
fs = 1.0 #30.0       
cutoff = 1/30 #3.667  

b, a = butter_lowpass(cutoff, fs, order)

# Plotting the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


# Creating the data for filteration
T = 300 #5.0         # value taken in seconds
n = int(T * fs) # indicates total samples
t = np.linspace(0, T, n, endpoint=False)

data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)

# Filtering and plotting
y = butter_lowpass_filter(data, cutoff, fs, order)
y_manual = test_butter_manual(data, cutoff, fs, order)
y_manual_2 = test_butter_manual_2(data, cutoff, fs, order)

#print('y - y_manual = ', repr(y - y_manual))
print('y - y_manual_2 = ', repr(y - y_manual_2))

plt.subplot(2, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.plot(t, y_manual_2, 'r--', linewidth=2, label='manually filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()
'''
