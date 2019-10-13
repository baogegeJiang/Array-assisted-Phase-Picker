import numpy as np
from numba import jit,float32, int64
import scipy.signal  as signal
nptype=np.float32
@jit
def xcorr(a,b):
    la=a.size
    lb=b.size
    c=np.zeros(la-lb+1)
    tb=0
    for i in range(lb):
        tb+=b[i]*b[i]
    for i in range(la-lb+1):
        ta=0
        tc=0
        ta= (a[i:(i+lb)]*a[i:(i+lb)]).sum()
        tc= (a[i:(i+lb)]*b[0:(0+lb)]).sum()
        if ta!=0 and tb!=0:
            c[i]=tc/np.sqrt(ta*tb)
    return c

@jit
def xcorrSimple(a,b):
    la=a.size
    lb=b.size
    c=np.zeros(la-lb+1)
    for i in range(la-lb+1):
        tc= (a[i:(i+lb)]*b[0:(0+lb)]).sum()
        c[i]=tc
    return c

@jit
def xcorrEqual(a,b):
    la=a.size
    lb=b.size
    c=np.zeros(la)
    tb0=(b*b).sum()
    for i in range(la):
        i1=min(i+lb,la)
        ii1=i1-i
        #print(ii1)
        tc= (a[i:i1]*b[0:ii1]).sum()
        tb=tb0
        if ii1!=lb:
            tb=(b[0:ii1]*b[0:ii1]).sum()
        c[i]=tc/np.sqrt(tb)
    return c

def corrNP(a,b):
    a=a.astype(nptype)
    b=b.astype(nptype)
    if len(b)==0:
        return a*0+1
    c=signal.correlate(a,b,'valid')
    tb=(b**2).sum()**0.5
    taL=(a**2).cumsum()
    ta0=taL[len(b)-1]**0.5
    taL=(taL[len(b):]-taL[:-len(b)])**0.5
    c[1:]=c[1:]/tb/taL
    c[0]=c[0]/tb/ta0
    return c,c.mean(),c.std()

@jit
def getDetec(x, minValue=0.2, minDelta=200):
    indexL = [-10000]
    vL = [-1]
    for i in range(len(x)):
        if x[i] <= minValue:
            continue
        if i > indexL[-1]+minDelta:
            vL.append(x[i])
            indexL.append(i)
            continue
        if x[i] > vL[-1]:
            vL[-1] = x[i]
            indexL[-1] = i
    if vL[0] == -1:
        indexL = indexL[1:]
        vL = vL[1:]
    return np.array(indexL), np.array(vL)

def matTime2UTC(matTime,time0=719529):
    return (matTime-time0)*86400

@jit(int64(float32[:],float32,int64,int64,float32[:]))
def cmax(a,tmin,winL,laout,aM):
    i=0 
    while i<laout:
        if a[i]>tmin:
            j=0
            while j<min(winL,i):
                if a[i]>a[i-j]:
                    a[i-j]=a[i]
                j+=1
        if i>=winL:
            aM[i-winL]+=a[i-winL]
        i+=1
    while i<laout+winL:
        aM[i-winL]+=a[i-winL]
        i+=1

def cmax_bak(a,tmin,winL,laout,aM):
    i=0 
    indexL=np.where(a>tmin)[0]
    for i in indexL:
        a[max(i-winL,0):i]=np.fmax(a[max(i-winL,0):i],a[i])
    aM[:laout]+=a[:laout]

def CEPS(x):
    #sx=fft(x);%abs(fft(x)).^2;
    #logs=log(sx);
    #y=abs(fft(logs(1:end)));
    spec=np.fft.fft(x)
    logspec=np.log(spec*np.conj(spec))
    y=abs(np.fft.ifft(logspec))
    return y