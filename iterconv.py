import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from math import cos,sin,asin,atan,tan
from mathFunc import xcorr,xcorrEqual,CEPS
from obspy import taup
from distaz import DistAz
import scipy.io as sio
import os

rad2deg=1/np.pi*180

def iterconv(L,Q,f=[1/10,4],f0=1,delta=0.02,N=2,threshold=0.01,\
    isAbs=True,isPlot=False,maxIT=100,isSave=False,isFFTShift=False\
    ,isAbsJust=False):
    if isFFTShift:
        L0=np.zeros(Q.size)
        L0[:L.size]=L
        L=L0
    fl=f[0]
    fh=f[1]
    wn0=fl/(0.5/delta)
    wn1=fh/(0.5/delta)
    print(wn0,wn1)
    b,a=signal.butter(4,[wn0,wn1],'bandpass')
    LL0=np.copy(L)
    QQ0=np.copy(Q)
    L=signal.filtfilt(b,a,L)
    Q=signal.filtfilt(b,a,Q)
    Q=Q/np.linalg.norm(L)
    L=L/np.linalg.norm(L)
    normQ0=np.linalg.norm(Q[:-int(L.size/3)])
    print('norm %.2f'%normQ0)
    wn=f0/(0.5/delta)
    b,a=signal.bessel(2,wn)
    L0=L
    Q0=Q
    Q=np.copy(Q)
    Q[:Q0.size]=Q0
    indexL=np.zeros(maxIT,dtype=np.int)
    aL=np.zeros(maxIT,dtype=np.float)
    count=0
    for i in range(maxIT):
        indexL[i],aL[i]=__interconv(L,Q,isAbs=isAbs,isFFTShift=isFFTShift)
        count=count+1
        if np.linalg.norm(Q[:])<threshold:
            break
    R=np.zeros(Q.size)
    for i in range(count):
        R[indexL[i]]=R[indexL[i]]+aL[i]
    Qout=np.convolve(L,R)
    Rout=signal.filtfilt(b,a,R)
    if isPlot:
        plt.subplot(2,2,1)
        plt.plot(L0,'b')
        plt.plot(L,'r')
        plt.xlim([0,len(Q)])
        plt.subplot(2,2,2)
        plt.plot(R,'b')
        plt.plot(Rout,'r')
        plt.xlim([0,len(Q)])
        plt.subplot(2,2,3)
        plt.plot(Q0,'b')
        plt.plot(Qout,'r')
        plt.plot(np.convolve(L,Rout),'g')
        plt.xlim([0,len(Q)])
        plt.subplot(2,2,4)
        plt.plot(Q)
        plt.xlim([0,len(Q)])
    if isAbsJust:
        Rout=Rout/np.sign(Rout[0]+0.00000001)
    if isSave:
        sio.savemat('tmpMat.mat',{'L':LL0,'Q':QQ0,'R':R,'Qout':Q,'QoutS':np.convolve(L,Rout)})
    return Rout,np.linalg.norm(Q[:-int(L.size/3)])/normQ0


def __interconv(L,Q,isAbs=True,isFFTShift=False):
    if isFFTShift:
        x=np.real(np.fft.ifft(np.conj(np.fft.fft(L))*(np.fft.fft(Q))))
        #plt.plot(x)
        #plt.show()
    else:
        x=xcorrEqual(Q,L)
    if isAbs:
        #-int(L.size/3)
        index=np.abs(x[:-int(L.size/3)]).argmax()
    else:
        index=x.argmax()
    a=x[index]
    if isFFTShift:
        Q[:]=Q[:]-a*timeShift(L,index)
    else:
        index1=min(index+L.size,x.size)
        Q[index:index1]=Q[index:index1]-a*L[:(index1-index)]
    return index,a

def timeShift(x,index):
    return np.fft.ifft(np.fft.fft(x)*np.exp(-np.arange(0,1,1/x.size)*2j*index*np.pi))

def test(N=100,M=200,delta=0.02,count=10,rq=0.1,rl=0.1,maxIT=100):
    L=np.random.rand(N)-0.5
    Q=np.random.rand(M)*(1-0.5)*rq
    R0=np.zeros(M)
    for i in range(10):
        index=int(np.random.rand(1)*(M-1))

        a=np.random.rand(1)-0.5
        R0[index]=a
        #print(index,a,(max(index+N,M)-index))
        Q[index:min(index+N,M)]+=L[:(min(index+N,M)-index)]*a
    
    L=L+np.random.rand(N)*rl+Q[:L.size]*rl
    L1=L.copy()
    Q1=Q.copy()
    plt.subplot(2,2,2)
    plt.plot(R0+1,'g')
    iterconv(L,Q,isPlot=True,maxIT=maxIT,delta=0.1,isFFTShift=False)
    plt.subplot(2,2,2)
    plt.plot(R0+1,'g')
    iterconv(L1,Q1,isPlot=True,maxIT=maxIT,delta=0.1,isFFTShift=True)

def calAI(inj,r=1.73):
    p=inj
    s=asin(sin(p)/r)
    return atan(2/tan(p)/(1/tan(s)*1/tan(s)-1))

def dConv(L,Q,eps=0.1,mod='fft',isPlot=False,delta=0.02,isFFTShift=False):
    if mod=='fft':
        specL=(np.fft.fft(L))
        specQ=(np.fft.fft(Q))
        specL=eps*np.abs(specL).max()+specL
        return np.fft.ifft(specQ/specL)
    elif mod =='ceps':
        return CEPS(L+1j*Q)
    elif mod == 'xcorr':
        QN=np.zeros(L.size+Q.size-1)
        QN[:Q.size]=Q
        return xcorr(QN,L)
    elif mod=='iterconv':
        N=int(L.size)
        return iterconv(L[:N],Q,isPlot=isPlot,delta=delta,isFFTShift=isFFTShift)

def receiverFunc(data,theta,AI,eps=0.05,mod='fft',isPlot=True,delta=0.02,isFFTShift=False):
    #LTQ
    data=data/(data*data).sum().sum()
    H=-(data[:,1]*cos(theta)+data[:,0]*sin(theta))
    L=H*sin(AI)+data[:,2]*cos(AI)
    Q=-H*cos(AI)+data[:,2]*sin(AI)
    index0=0
    index1=len(L)
    indexL=np.arange(index0,index1)
    if isPlot and mod != 'iterconv':
        plt.subplot(4,1,2)
        plt.plot(np.abs(np.fft.fft(L/np.linalg.norm(L))),'b')
        plt.plot(np.abs(np.fft.fft(Q/np.linalg.norm(Q))),'r')
        plt.subplot(4,1,1)
        plt.plot(((L/np.linalg.norm(L))),'b')
        plt.plot(((Q/np.linalg.norm(L))),'r')
        plt.subplot(4,1,3)
        x=xcorr(Q,L[indexL])
        plt.plot(np.arange(x.size)-index0,x)
    return np.real(dConv(L,Q,eps=eps,mod=mod,isPlot=isPlot,delta=delta,\
        isFFTShift=isFFTShift))

def calPhiTheta(v):
    v=np.array(v)
    e=v[0]
    n=v[1]
    z=v[2]
    h=np.sqrt(n*n+e*e)
    phi=atan(h/z)
    theta=atan(e/n)+np.pi
    return phi,theta

def calConvPhiTheta(w,isAbsJust=True):
    m=calConvM(w)
    b,v=np.linalg.eig(m/m.sum())
    maxIndex=b.argmax()
    vMax=v[:,maxIndex]
    if isAbsJust:
        vMax=vMax/np.sign(vMax[-1])
    return calPhiTheta(vMax)

model0='iasp91'
taupM0=taup.TauPyModel(model=model0)

def rfOnQuake(quake,waveform,staInfos,mod='fft',r=1.73,\
    isPlot=False,indexL=np.arange(-10,15*50),staRF=[],\
    staNum=None,model=None,f=[1/20,2],isFFTShift=False,\
    outDir='rfFig/',isPrint=True):
    count=0
    if model == None:
        taupM=taupM0
    else:
        taupM=taup.TauPyModel(model=model)
    for record in quake:
        delta=waveform['deltaL'][0,count]
        fl=f[0]
        fh=f[1]
        if delta ==0:
            print('bad waveform')
            count+=1
            continue
        wn0=fl/(0.5/delta)
        wn1=fh/(0.5/delta)
        b,a=signal.butter(2,[wn0,wn1],'bandpass')
        staIndex=record.getStaIndex()
        staInfo=staInfos[staIndex]
        if True:
            dis=DistAz(quake.loc[0],quake.loc[1],staInfo['la'],\
                staInfo['lo'])
            BAZ=dis.getBaz()/180*np.pi-staInfo['az']/180*np.pi
            Dis=dis.getDelta()
            ray=taupM.get_travel_times(max(min(1,quake.loc[2]+staInfo['dep']/1000),100),\
                Dis,phase_list=('p','P'))[0]
            inj=ray.incident_angle/180*np.pi
            AI=calAI(inj,r=r)*0
        index0=np.where(waveform['indexL'][0][:]==0)[0][0]
        w=waveform['pWaveform'][count,indexL+index0,:]
        w[:,2]=-w[:,2]
        pre=waveform['pWaveform'][count,\
            max(0,indexL[0]+index0-1000):indexL[0]+index0-2,:]
        for comp in range(3):
            w[:,comp]=signal.filtfilt(b,a,w[:,comp])
            pre[:,comp]=signal.filtfilt(b,a,pre[:,comp])
        wMax=np.abs(w).max()
        preMax=np.abs(pre).max()
        if wMax.max()<=preMax.max()*3 or wMax.max()==0:
            print('bad waveform')
            count+=1
            continue
        
        AI0,BAZ0=calConvPhiTheta(w)
        print("%.2f %.2f"%(AI*rad2deg,BAZ*rad2deg))
        rf,rms=receiverFunc(w,BAZ,\
            AI*0,mod=mod,isPlot=isPlot,delta=delta,isFFTShift=isFFTShift)
        print(rms)
        if rf.any()==None or rf.any()==np.nan:
            return None
        if len(staRF)!=0 and rms<0.15:
            staRF[staIndex,:]+=rf
            staNum[staIndex]+=1
            
        theta=-BAZ
        phi=AI
        if isPlot and mod!='iterconv':
            plt.subplot(4,1,4)
            plt.plot((indexL-float(indexL[0]))*float(waveform['deltaL'][0,count]),rf)
            
            plt.show()
        count+=1
        if isPrint and isPlot:
            staDir=outDir+'/'+str(staIndex)+'/'
            if not os.path.exists(outDir):
                os.mkdir(outDir)
            if not os.path.exists(staDir):
                os.mkdir(staDir)
            filename=staDir+os.path.basename(quake.filename)[:-4]+str(staIndex)+'.png'
            #plt.title
            plt.suptitle(('%.2f'%rms)+' '+str(quake.loc[0])+' '+str(quake.loc[1])+' '+('%.2f'%(BAZ/np.pi*180))+' '+('%.2f'%(BAZ0/np.pi*180)))
            plt.savefig(filename)
            plt.close()
        elif isPlot:
            plt.show()
            

def calConvM(data):
    dataMat=np.mat(data)
    return dataMat.transpose()*dataMat

def rfOnQuakeL(quakeL,waveformL,staInfos,mod='fft',indexL=np.arange(-100,250),isPlot=False):
    staN=len(staInfos)
    N=len(waveformL)
    staNum=np.zeros(staN)
    staRF=np.zeros((staN,indexL.size))
    for i in range(N):
        rfOnQuake(quakeL[i],waveformL[i],staInfos,indexL=indexL\
            ,staRF=staRF,staNum=staNum,mod=mod,isPlot=isPlot)
    return staRF,staNum

def showStaRf(staRF,staNum,staInfos,R,Az0=0,dAz0=10/360*np.pi,minNum=10):
    midLa=(R[0]+R[1])/2
    midLo=(R[2]+R[3])/2
    for i in range(len(staInfos)):
        if staNum[i]<minNum:
            continue
        dis=DistAz(midLa,midLo,staInfos[i]['la'],staInfos[i]['lo'])
        Az=dis.getAz()/180*np.pi
        delta=dis.getDelta()
        dAz=Az-Az0
        if sin(np.abs(dAz))>sin(dAz):
            continue
        delta=delta*cos(dAz)
        plt.plot(staRF[i,:]/staNum[i]*1+delta,'b')
    plt.show()