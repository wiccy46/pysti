from scipy.signal import butter, lfilter, firwin
from scipy.interpolate import interp1d
from warnings import catch_warnings,simplefilter
import numpy as np

def octave_filters(x, fs, N=6, f0 =[125, 250, 500, 1000, 2000, 4000, 8000],
                   hammingTime=16.6):
    """Octave band filters for STI measurement. 
    
    Parameters
    ----------
    fs : int
        Sampling frequency
    N : int
        Filter order
    f0 : list
        Center frequency of each octave
    """
    # calculate the nyquist frequency
    nyquist = fs * 0.5
    # length of Hamming window for FIR low-pass at 25 Hz
    # hammingLength = (hammingTime / 1000.0) * fs
    # process each octave band
    for f in f0:
        # filter the output at the octave band f
        lowcut = f / np.sqrt(2)  # lowcut
        highcut = f * np.sqrt(2)
        if f < max(f0):
            with catch_warnings():      # suppress the spurious warnings given
                simplefilter('ignore')  # under certain conditions
                b1, a1 = butter(N, lowcut / nyquist, btype='high')
                b2, a2 = butter(N, highcut / nyquist, btype='low')
            filtOut = lfilter(b1, a1, x)   # high-pass raw audio at f1
            filtOut = lfilter(b2, a2, filtOut)  # low-pass after high-pass at f1
        else:
            with catch_warnings():
                simplefilter('ignore')
                b1, a1 = butter(N, f / nyquist, btype='high')
            filtOut = lfilter(b1, a1, x)
#         filtOut = np.array(filtOut) ** 2
#         b = firwin(int(hammingLength), 25.0, nyq=nyquist)
#         filtOut = lfilter(b, 1, filtOut)
#         filtOut = filtOut * -1.0
        # stack-up octave band filtered audio
        try:
            filtered_audio = np.vstack((filtered_audio, filtOut))
        except:
            filtered_audio = filtOut
    return filtered_audio


def pow2db(x):
    return 10 * np.log10(x)


def get_mtf(ir):
    return np.abs(np.fft.rfft(ir ** 2) / np.sum(ir ** 2))


def sti(ir, fs):
    
    # 1. filter ir through octave bands
    # 2. get MTF through 
    Nfc = 7  # 7 octaves from 125 - 8000Hz
    
    
    ir_filtered = octave_filters(ir, fs)
    # MTF function. 
    mtf = get_mtf(ir_filtered)
    
    modulation_freqs = [0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50, 3.15,
                        4.00, 5.00, 6.30, 8.00, 10.00, 12.50]
    freqs = np.linspace(0, fs // 2, mtf.shape[1])
    # freqs[-1] = 0 # No nyquist frequency

    m = np.zeros((len(modulation_freqs), Nfc))
    for i in range(Nfc):
        # Old x is freqs, y is mtf[i, :], newx = modulation_freqs
        # m(i,:) = interp1(freqs,MTF_octband(1:end/2,i),modulation_freqs);
        interp = interp1d(freqs, mtf[i, :])
        m[:, i] = interp(modulation_freqs)
    

    # Convert each of the 98m values into an apparent SNR in dB
    SNR_apparent = pow2db(m / (1 - m))
    
    # Limit the range
    SNR_apparent_clipped = np.clip(SNR_apparent, -15, 15)

    
    # Compute the mean (S/N) for each octave band
    SNR_avg = np.mean(SNR_apparent_clipped, axis=0)
    
    # weight the octave mean (S/N) vals
    W = [0.13, 0.14, 0.11, 0.12, 0.19, 0.17, 0.14]
    
    weighted_m = m * W

    SNR_avg_weighted = np.sum(np.dot(weighted_m, SNR_avg))
    
    
    SNR_avg_weighted_approx = np.sum(np.dot(weighted_m / np.sum(weighted_m),
                                            SNR_avg))
    
    # Convert the overall mean to an STI val
    sti_val = (SNR_avg_weighted + 15) / 30
    sti_val_approx = (SNR_avg_weighted_approx + 15) / 30
    
    return (sti_val, sti_val_approx)