import librosa
import numpy as np
from scipy.fftpack import dct
import sys

# If you want to see the spectrogram picture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot_spectrogram(spec, note,file_name):
    """Draw the spectrogram picture
        :param spec: a feature_dim by num_frames array(real)
        :param note: title of the picture
        :param file_name: name of the file
    """
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.savefig(file_name)


#preemphasis config 
alpha = 0.97

# Enframe config
frame_len = 400      # 25ms, fs=16kHz
frame_shift = 160    # 10ms, fs=15kHz
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12

# Read wav file
wav, fs = librosa.load('./test.wav', sr=None)

# pre-emphasis  function
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def _my_hamming(frame_len):
    n=np.array(range(frame_len))
    if frame_len>1:
        return 0.54-0.46*np.cos(2*np.pi*n/(frame_len-1))
    elif frame_len==1:
        return np.ones(1,float)
    else:
        return np.array([])


# Enframe with Hamming window function
def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """
    
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift)+1
    frames = np.zeros((int(num_frames),frame_len))
    for i in range(int(num_frames)):
        frames[i,:] = signal[i*frame_shift:i*frame_shift + frame_len] 
        # frames[i,:] = frames[i,:] * win
        frames[i,:] = frames[i,:] * _my_hamming(frame_len)

    return frames

def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2 ) + 1
    spectrum = np.abs(cFFT[:,0:valid_len])
    # print(cFFT,cFFT.shape)  #(356, 512)
    # print(spectrum,spectrum.shape)  #(356, 257)
    return spectrum

def fbank(spectrum = None, num_filter = num_filter,fs=fs):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """
    #此函数内容参考：http://fancyerii.github.io/books/mfcc/

    # print(spectrum,spectrum.shape,fs)    #(356, 257) 16000
    #(356, 257) -> (356, 23)

    #功率谱
    pow_frames=1/fft_len*np.square(spectrum)
    # print(pow_frames,pow_frames.shape)  #(356, 257)

    feats=np.zeros([spectrum.shape[0], num_filter])

    fh=fs/2
    num_fft=spectrum.shape[1]   #257
    mel_fl=0
    mel_fh=2595*np.log10(1+fh/700)
    mel_points=np.linspace(mel_fl,mel_fh,num_filter+2)  #按滤波器个数分成n个长度一样的滤波器
    print(mel_points,len(mel_points))   #长度为25的在mel频域上的等差数列，即平均分为23个滤波器
    # sys.exit()

    #还应把mel频率转换为标准频率
    hz_points=700*(10**(mel_points/2595)-1)
    print(hz_points,hz_points.shape)    #(25,),这是按mel频率划分的23个滤波器的频率点

    bin=np.floor((fft_len+1)*hz_points/fs)     #因为FFT的频率没办法精确的与上面的频率对应，因此需把它们对应到最近的bin里面
    print(bin,bin.shape)    #(25,)
    #[  0.   2.   5.   8.  11.  15.  19.  24.  29.  35.  41.  48.  56.  65.  75.  85.  97. 111. 126. 142. 160. 181. 203. 228. 256.]

    fb=np.zeros([num_filter,int(np.floor(num_fft))])    #设计滤波器的形状为（23，257）
    # print(fb,fb.shape)      #(23, 257)
    # sys.exit()

    for m in range(1,num_filter+1): #遍历23个滤波器
        start=bin[m-1]
        mid=bin[m]
        end=bin[m+1]
        for k in range(int(start),int(mid)):
            fb[m-1,k]=(k-start)/(mid-start)
        for k in range(int(mid),int(end)):
            fb[m-1,k]=(end-k)/(end-mid)
    print(fb,fb.shape)  #fb就为设计好的mel滤波器

    # filter_banks=np.dot(spectrum,fb.T)    #输入特征与mel滤波器进行卷积（矩阵乘法）
    filter_banks=np.dot(pow_frames,fb.T)    #这里用功率谱与mel滤波器相乘

    feats=np.log(filter_banks)          #再取对数，就得FBank特征
    print(filter_banks,filter_banks.shape)  #(356, 23)
    print(feats,feats.shape)

    return feats

def mfcc(fbank, num_mfcc = num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """

    feats = np.zeros((fbank.shape[0],num_mfcc))
    # num_fbank=fbank.shape[1]
    mfcc = dct(fbank, type=2, axis=1, norm='ortho')[:, 1 : (num_mfcc + 1)]
    # print(mfcc.shape)   #(356, 12)

    return mfcc

def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f=open(file_name,'w')
    (row,col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i,j])+' ')
        f.write(']\n')
    f.close()


def main():
    wav, fs = librosa.load('./test.wav', sr=None)
    signal = preemphasis(wav)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)
    fbank_feats = fbank(spectrum=spectrum,num_filter = num_filter,fs=fs)
    mfcc_feats = mfcc(fbank_feats)

    # plot_spectrogram(fbank_feats.T, 'Filter Bank','fbank.png')
    # write_file(fbank_feats,'./test.fbank')
    # plot_spectrogram(mfcc_feats.T, 'MFCC','mfcc.png')
    # write_file(mfcc_feats,'./test.mfcc')

if __name__ == '__main__':
    main()
