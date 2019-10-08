import numpy as np

def CEPS(x):
    #sx=fft(x);%abs(fft(x)).^2;
    #logs=log(sx);
    #y=abs(fft(logs(1:end)));
    spec=np.fft.fft(x)
    logspec=np.log(spec*np.conj(spec))
    y=abs(np.fft.ifft(logspec))
    return y
