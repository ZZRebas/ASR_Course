import numpy as np
import sys


# sys.exit()

# a=np.array(range(1,10))
# print(a)
# b=a[1:]-a[:-1]
# c=np.append(a[0],a[1:]-0.9*a[:-1])
# print(c)
# print(c.shape)
# signal=c
# num_samples = signal.size
# num_frames = np.floor((num_samples - 25*16) / 10*16)+1



signal=np.array([ 0.08      ,  0.15302337,  0.34890909,  0.60546483,  0.84123594, # may vary
               0.98136677,  0.98136677,  0.84123594,  0.60546483,  0.34890909,
               0.15302337,  0.08      ])

import matplotlib.pyplot as plt
from numpy.fft import fft,fftshift
window=np.hamming(51)
plt.plot(window)
plt.title("Hamming window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
# plt.show()

plt.figure()
A=fft(window,2048)/25.5
mag=np.abs(fftshift(A))
freq=np.linspace(-0.5,0.5,len(A))
response=20*np.log10(mag)
response=np.clip(response,-100,100)
plt.plot(freq,response)
plt.title("Frequency response of Hamming window")
plt.ylabel("Magnitude [dB]")
plt.xlabel("Normalized frequency [cycles per sample]")
plt.axis('tight')
# plt.show()


def my_hamming(frame_len):
    n=np.array(range(frame_len))
    if frame_len>1:
        return 0.54-0.46*np.cos(2*np.pi*n/(frame_len-1))
    elif frame_len==1:
        return np.ones(1,float)
    else:
        return np.array([])
win=my_hamming(10)
# print(win)
# print(win.shape)
x=range(10)
# plt.plot(x,win)
# plt.show()


