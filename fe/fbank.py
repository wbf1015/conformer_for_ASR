import librosa
import os
import sys
import argparse
import numpy as np
import matplotlib
import scipy.io.wavfile as wav
from scipy.fftpack import dct
from python_speech_features import logfbank

sys.path.append('../')
from config import get_fbank_dim
# If you want to see the spectrogram picture




matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_spectrogram(spec, note, file_name):
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


# preemphasis config
alpha = 0.97

# Enframe config
frame_len = 400  # 25ms, fs=16kHz
frame_shift = 160  # 10ms, fs=15kHz
fft_len = 512

# Mel filter config
num_filter = get_fbank_dim()
num_mfcc = 12

# read_ground_truth
ground_truth_list = []

# the_max_len to padding 0
max_len = 2048

# Enframe with Hamming window function
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """

    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift) + 1
    frames = np.zeros((int(num_frames), frame_len))
    for i in range(int(num_frames)):
        frames[i, :] = signal[i * frame_shift:i * frame_shift + frame_len]
        frames[i, :] = frames[i, :] * win

    return frames


def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2) + 1
    spectrum = np.abs(cFFT[:, 0:valid_len])
    return spectrum


def fbank(fs, spectrum, num_filter=num_filter):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """
    low_mel_freq = 0
    high_mel_freq = 2595 * np.log10(1 + (fs / 2) / 700)  # 转到梅尔尺度上
    mel_filters_points = np.linspace(low_mel_freq, high_mel_freq, num_filter + 2)
    freq_filters_pints = (700 * (np.power(10., (mel_filters_points / 2595)) - 1))
    freq_bin = np.floor(freq_filters_pints / (fs / 2) * (fft_len / 2 + 1))
    feats = np.zeros((int(fft_len / 2 + 1), num_filter))
    for m in range(1, num_filter + 1):
        bin_low = int(freq_bin[m - 1])
        bin_medium = int(freq_bin[m])
        bin_high = int(freq_bin[m + 1])
        for k in range(bin_low, bin_medium):
            feats[k, m - 1] = (k - freq_bin[m - 1]) / (freq_bin[m] - freq_bin[m - 1])
        for k in range(bin_medium, bin_high):
            feats[k, m - 1] = (freq_bin[m + 1] - k) / (freq_bin[m + 1] - freq_bin[m])
    feats = np.dot(spectrum, feats)
    feats = 20 * np.log10(feats)
    return feats


def mfcc(fbank, num_mfcc=num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array
    """

    # feats = np.zeros((fbank.shape[0],num_mfcc))
    mfcc = dct(fbank, type=2, axis=1, norm='ortho')[:, 1:(num_mfcc + 1)]
    return mfcc


def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f = open(file_name, 'w')
    (row, col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i, j]) + ' ')
        f.write(']\n')
    f.close()


def get_audio_feature(path, file_type):
    # wav, fs = librosa.load(path, sr=None)
    # signal = preemphasis(wav)
    # frames = enframe(signal)
    # spectrum = get_spectrum(frames)
    # fbank_feats = fbank(fs, spectrum)
    # 加载音频文件
    (rate, signal) = wav.read(path)
    # 提取fbank特征
    fbank_feats = logfbank(signal, rate, nfilt=80)
    global max_len
    target_shape=(max_len, num_filter)
    expanded_feats = np.zeros(target_shape)
    expanded_feats[:fbank_feats.shape[0], :] = fbank_feats
    fbank_feats = expanded_feats
    
    if file_type == 'test':
        save_path = '/root/autodl-tmp/AIshell1_test/' + path[-20:-4]
    if file_type == 'train':
        save_path = '/root/autodl-tmp/AIshell1_train/' + path[-20:-4]
    np.save(save_path, fbank_feats)


def get_audio_features_train(start, end, file_type):
    for index in range(start, end+1):
        if file_type == 'train':
            base_path = '/root/autodl-tmp/train/'
        if file_type == 'test':
            base_path = '/root/autodl-tmp/test/'
        folder_name = ''
        if 2<=index<= 9:
            folder_name = 'S000'+str(index)
        if 10<=index<=99:
            folder_name = 'S00'+str(index)
        if 100<=index:
            folder_name ='S0'+str(index)
        
        base_path = base_path + folder_name + '/'
        file_list = os.listdir(base_path)
        
        for file in file_list:
            wav_path = base_path + file
            file_index = file[:-4]
            if has_ground_truth(file_index):
                get_audio_feature(wav_path, file_type)

        print(index,'文件夹提取特征成功')
        
            
def readFBankData(path):
    fbank = np.load(path, allow_pickle=True)
    return fbank           
            

def has_ground_truth(index):
    global ground_truth_list
    if len(ground_truth_list)==0:
         f = open('/root/autodl-tmp/AIshell1_truth.txt', encoding='utf-8')
         for line in f:
            s = line.strip()
            key = s[:16]
            ground_truth_list.append(key)
    if index in ground_truth_list:
        return True
    else:
        return False
            
            
        
def main():
    pass
    # wav, fs = librosa.load('/root/autodl-tmp/HLT/data/test.wav', sr=None)
    # signal = preemphasis(wav)
    # frames = enframe(signal)
    # spectrum = get_spectrum(frames)
    # fbank_feats = fbank(fs, spectrum)
    # print(fbank_feats.shape)
    # print(type(fbank_feats))
    # mfcc_feats = mfcc(fbank_feats)
    # plot_spectrogram(fbank_feats, 'Filter Bank','fbank.png')
    # write_file(fbank_feats,'./test.npy')
    # np.save('test.npy', fbank_feats)
    # plot_spectrogram(mfcc_feats.T, 'MFCC','mfcc.png')
    # write_file(mfcc_feats,'./test.mfcc')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature extracting .py")
    parser.add_argument('--start', type=int, help='start folder index', default=912)
    parser.add_argument('--end', type=int, help='end folder index', default=916)
    parser.add_argument('--type', type=str, help='train/test/dev', default='test')
    args = parser.parse_args()
    get_audio_features_train(args.start, args.end, args.type)
    # readFBankData('/root/autodl-tmp/AIshell1_train/BAC009S0002W0197.npy')
    


    
    


