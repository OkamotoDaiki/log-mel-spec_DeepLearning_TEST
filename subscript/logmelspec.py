import numpy as np
from subscript import ConvertHzMel

def get_logmelspec(data, fs):
    """
    Calculation of log_mel spectrum.

    Attributes:
        data: time series data.

    Returns:
        log_mel_spec: log_mel spectrum data.
    """
    convert_hz_mel = ConvertHzMel.ConvertHzMel()
    N = len(data)
    fscale = np.linspace(0, fs, N)[:int(N/2)]
    mel_scale = convert_hz_mel.hz2mel(fscale)

    dft = np.abs(np.fft.fft(data))[:int(N/2)]
    mel_spec = convert_hz_mel.hz2mel(dft)
    log_mel_spec = 10*np.log10(mel_spec**2)
    return log_mel_spec, mel_scale


if __name__=="__main__":
    """plot graph"""
    import soundfile
    import matplotlib.pyplot as plt
    fname = 'recordings/0_jackson_0.wav' #any wav file.
    data, fs = soundfile.read(fname)
    logmelspec_array, mel_scale = get_logmelspec(data, fs)
    
    plt.subplot(121)
    plt.plot(mel_scale, logmelspec_array)
    plt.xlabel("Mel")
    plt.ylabel("log amplitude spectrum [dB]")

    #DFT plot
    N = len(data)
    dft = np.abs(np.fft.fft(data))[:int(N/2)]
    spec = 10*np.log10(dft**2)
    fscale = np.linspace(0, fs, N)[:int(N/2)]
    
    plt.subplot(122)
    plt.plot(fscale, spec)
    plt.xlabel("frequency[Hz]")
    png_fname = "comparison_mel_Hz.png"
    plt.savefig(png_fname)
    plt.show()