import numpy as np
from numba import jit, float32, int64
from obspy import UTCDateTime
from tool import getDetec, QuakeCC, RecordCC
import matplotlib.pyplot as plt
import time as Time
from distaz import DistAz
from multiprocessing import Process, Manager
import scipy.signal  as signal
import cudaFunc
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)
maxStaN=20
nptype=np.float32
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

corrTorch=cudaFunc.torchcorrnn

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



def getTimeLim(staL):
    n = len(staL)
    bTime = UTCDateTime(1970,1,1).timestamp
    eTime = UTCDateTime(2200,12,31).timestamp
    for i in range(n):
        bTime = max([staL[i].bTime, bTime])
        eTime = min([staL[i].eTime, eTime])
    return bTime, eTime

def doMFT(staL, waveform,bTime, n, wM=np.zeros((2*maxStaN,86700*50),dtype=nptype),delta=0.02\
        ,minMul=3,MINMUL=8, winTime=0.4,minDelta=20*50, locator=None,tmpName=None,quakeRef=None,\
        maxCC=1,R=[-90,90,-180,180],staInfos=None,maxDis=200,deviceL=['cuda:0']):
    time_start = Time.time()
    winL=int(winTime/delta)
    if waveform['pTimeL'].size<5:
        return []
    staSortL = np.argsort(waveform['pTimeL'][0])
    tmpTimeL= np.arange(-1,3,delta).astype(nptype)
    tmpRefTime=waveform['indexL'][0][0]*delta
    tmpIndexL=((tmpTimeL-tmpRefTime)/delta).astype(np.int64)
    aM=torch.zeros(n,device=deviceL[0])
    staIndexL=[]
    staIndexOL=[]
    phaseTypeL=[]
    mL=[]
    sL=[]
    oTimeL=[]
    oTime=waveform['time']
    index=0
    pCount=0
    for i in range(staSortL.size):
        staIndex=staSortL[i]
        staIndexO=int(waveform['staIndexL'][0][staIndex])
        if staIndexO>=len(staL):
            continue
        if staInfos!=None:
            staInfo=staInfos[staIndexO]
            if staInfo['la']<R[0] or \
                staInfo['la']>R[1] or \
                staInfo['lo']<R[2] or \
                staInfo['lo']>R[3]:
                continue
        if quakeRef !=None and staInfos !=None:
            staInfo=staInfos[staIndexO]
            dis=DistAz(quakeRef.lco[0],quakeRef.loc[1],staInfo['la'],staInfo['lo'])
            if dis.degreesToKilometers(dis.getDelta())>maxDis:
                continue
        if waveform['pTimeL'][0][staIndex]!=0 and staL[staIndexO].data.data.shape[-1]>1000:
            dTime=(waveform['pTimeL'][0][staIndex]-oTime+bTime-staL[staIndexO].data.bTime.timestamp)
            dIndex=int(dTime/delta)
            if dIndex<0:
                continue
            c,m,s=corrTorch(staL[staIndexO].data.data[2,:],waveform['pWaveform'][staIndex,tmpIndexL,2])
            if s==1:
                continue
            staIndexL.append(staIndex)
            staIndexOL.append(staIndexO)
            phaseTypeL.append(1)
            oTimeL.append(dTime+staL[staIndexO].data.bTime.timestamp-tmpTimeL[0])
            mL.append(m)
            sL.append(s)
            wM[index]=torch.zeros(n+50*100,device=c.device)
            wM[index][0:c.shape[0]-dIndex]=c[dIndex:]
            threshold=m+minMul*s
            cudaFunc.torchMax(c[dIndex:],threshold,winL, aM)

            index+=1
            pCount+=1
            if pCount>=maxStaN:
                break
        if waveform['sTimeL'][0][staIndex]!=0 and staL[staIndexO].data.data.shape[-1]>1000:
            dTime=(waveform['sTimeL'][0][staIndex]-oTime+bTime-staL[staIndexO].data.bTime.timestamp)
            dIndex=int(dTime/delta)
            if dIndex<0:
                continue
            chanelIndex=0
            if waveform['sWaveform'][:,1].max()>waveform['sWaveform'][:,0].max():
                chanelIndex=1
            c,m,s=corrTorch(staL[staIndexO].data.data[chanelIndex,:],waveform['sWaveform'][staIndex,tmpIndexL,chanelIndex])
            if s==1:
                continue
            staIndexL.append(staIndex)
            staIndexOL.append(staIndexO)
            phaseTypeL.append(2)
            oTimeL.append(dTime+staL[staIndexO].data.bTime.timestamp-tmpTimeL[0])
            mL.append(m)
            sL.append(s)
            wM[index]=torch.zeros(n+50*100,device=c.device)
            wM[index][0:c.shape[0]-dIndex]=c[dIndex:]
            threshold=m+minMul*s
            cudaFunc.torchMax(c[dIndex:],threshold,winL, aM)
            index+=1
    if index<5:
        return []
    aM=aM/index
    M=aM[aM!=0].mean().cpu().numpy()
    S=aM[aM!=0].std().cpu().numpy()
    if S<5e-3:
        return []
    threshold=min(maxCC,M+MINMUL*S)
    indexL, vL= getDetec(aM.cpu().numpy(), minValue=threshold, minDelta=minDelta)
    print("M: %.5f S: %.5f thres: %.3f peakNum:%d"%(M,S,threshold,len(indexL)))
    wLL=np.arange(-10,winL)
    quakeL=[]
    print('corr',Time.time()-time_start)
    for i in range(len(indexL)):
        cc=vL[i]
        index = indexL[i]
        if index+wLL[0]<0:
            print('too close to the beginning')
            continue
        time= index*delta+bTime
        staD={}
        quakeCC = QuakeCC(cc,M,S,loc=waveform['loc'][0], time=time, tmpName=tmpName)
        phaseCount=0
        for j in range(len(staIndexL)):
            staIndexO=staIndexOL[j]
            dIndex=wM[j][index+wLL].argmax().cpu().numpy()
            phaseTime=float(oTimeL[j]+(wLL[dIndex]+index)*delta)
            if phaseTypeL[j]==1:
                quakeCC.append(RecordCC(staIndexO, phaseTime, 0,wM[j][index+wLL[dIndex]].cpu().numpy()\
                    , 0, mL[j], sL[j], 0, 0))
                staD[staIndexO]=phaseCount
                phaseCount+=1
            if phaseTypeL[j]==2:
                j0=staD[staIndexO]
                quakeCC[j0][2]=phaseTime
                quakeCC[j0][4]=wM[j][index+wLL[dIndex]].cpu().numpy()
                quakeCC[j0][7]=mL[j]
                quakeCC[j0][8]=sL[j]
        if locator != None and len(quakeCC)>=3:
            if quakeRef==None:
                quakeCC,res=locator.locate(quakeCC)
            else:
                quakeCC,res=locator.locateRef(quakeCC,quakeRef)
            print(quakeCC.time,quakeCC.loc,res,quakeCC.cc)
            if False:
                try:
                    if quakeRef==None:
                        quakeCC,res=locator.locate(quakeCC)
                    else:
                        quakeCC,res=locator.locateRef(quakeCC,quakeRef)
                    print(quakeCC.time,quakeCC.loc,res,quakeCC.cc)
                except:
                    print('wrong in locate')
                else:
                    pass
        quakeL.append(quakeCC)
    time_end=Time.time()
    print(time_end-time_start)
    return quakeL

def doMFTAll(staL,waveformL,bTime,n=86400*50,delta=0.02\
        ,minMul=4,MINMUL=8, winTime=0.4,minDelta=20*50, \
        locator=None,tmpNameL=None, isParallel=False,\
        NP=2,quakeRefL=None,maxCC=1,R=[-90,90,-180,180],\
        maxDis=200,isUnique=True,isTorch=True,deviceL=['cuda:0']):
    if not isParallel:
        quakeL=[]
        wM=[None for i in range(maxStaN*2)]
        count=0
        for waveform in waveformL:
            waveform['pWaveform']=waveform['pWaveform'].astype(np.float32)
            waveform['sWaveform']=waveform['sWaveform'].astype(np.float32)
        for sta in staL:
            if sta.data.data.shape[0]>1/delta or sta.data.data.shape[-1]>1/delta:
                if sta.data.data.shape[0]>sta.data.data.shape[-1]:
                    sta.data.data=sta.data.data.transpose()
                if  isTorch and (not isinstance(sta.data.data,torch.Tensor)):
                    count+=1
                    sta.data.data=torch.tensor(sta.data.data,device=deviceL[(count)%len(deviceL)])
            if sta.data.data.shape[-1]>22*3600/delta:
                bTime=max(bTime, sta.data.bTime.timestamp+1)
        for i in range(len(waveformL)):
            print('doing on %d'%i)
            if tmpNameL!=None:
                tmpName=tmpNameL[i]
            else:
                tmpName=None
            quakeRef=None
            if quakeRefL!=None:
                quakeRef=quakeRefL[i]
            quakeL=quakeL+doMFT(staL,waveformL[i],bTime,n,wM=wM,delta=delta,minMul=minMul,MINMUL=MINMUL,\
                winTime=winTime, minDelta=minDelta,locator=locator,tmpName=tmpName, quakeRef=quakeRef,\
                maxCC=maxCC,R=R,maxDis=200,deviceL=deviceL)
        if isUnique:
            quakeL=uniqueQuake(quakeL)
        for sta in staL:
            if sta.data.data.shape[0]>1/delta or sta.data.data.shape[-1]>1/delta:
                if isinstance(sta.data.data,torch.Tensor):
                    sta.data.data=sta.data.data.cpu().numpy()
                if sta.data.data.shape[0]<sta.data.data.shape[-1]:
                    sta.data.data=sta.data.data.transpose()
                
        return quakeL
    else:
        manager=Manager()
        staLP=[]#manager.list()
        staLP.append(staL)
        waveformLP=[]#manager.list()
        waveformLP.append(waveformL)
        quakeLs=[manager.list() for i in range(NP)]
        processes=[]
        for i in range(NP):
            process=Process(target=__doMFTAll,args=(\
                staLP,waveformLP,bTime,quakeLs[i],n,delta,\
                minMul,MINMUL,winTime,minDelta,locator,tmpNameL,NP,i))
            process.start()
            processes.append(process)
        for process in processes:
            print(process)
            process.join()
        quakeL=[]
        for quakeLTmp in quakeLs:
            quakeL=quakeL+quakeLTmp[0]
        return uniqueQuake(quakeL)


def __doMFTAll(staLP,waveformLP,bTime,quakeLP,n=86400*50,delta=0.02\
        ,minMul=4,MINMUL=8, winTime=0.4,minDelta=20*50, \
        locator=None,tmpNameL=None,NP=2,IP=0):
    staL=staLP[0]
    waveformL=waveformLP[0]
    quakeL=[]
    wM=np.zeros((2*maxStaN,n+50*100),dtype=nptype)
    for i in range(IP,len(waveformL),NP):
        print('doing on %d'%i)
        if tmpNameL!=None:
            tmpName=tmpNameL[i]
        else:
            tmpName=None
        quakeL=quakeL+doMFT(staL,waveformL[i],bTime,n,wM=wM,delta=delta,minMul=minMul,MINMUL=MINMUL,\
             winTime=winTime, minDelta=minDelta,locator=locator,tmpName=tmpName)
    quakeLP.append(quakeL)

def uniqueQuake(quakeL,minDelta=5, minD=0.3):
    PS=np.zeros((len(quakeL),7))
    for i in range(len(quakeL)):
        PS[i,0]=i
        PS[i,1]=quakeL[i].time
        PS[i,2:3]=quakeL[i].loc[0:1]
        PS[i,4]=quakeL[i].getMul()
        PS[i,5]=quakeL[i].cc
        PS[i,6]=quakeL[i].M
    L=np.argsort(PS[:,1])
    PS=PS[L,:]
    L=uniquePS(PS,minDelta=minDelta,minD=minD)
    quakeLTmp=[]
    for i in L:
        quakeLTmp.append(quakeL[i])
    return quakeLTmp


@jit
def uniquePS(PS,minDelta=20, minD=0.5):
    L=[]
    N=len(PS[:,0])
    for i in range(N) :
        isMax=1
        if np.isnan(PS[i,5]) or np.isnan(PS[i,6]):
            continue
        for j in range(i-1,0,-1):
            if np.isnan(PS[j,5]):
                continue
            if PS[j,1]<PS[i,1]-minDelta:
                break
            if np.linalg.norm(PS[j,2:3]-PS[i,2:3])>minD:
                continue
            if PS[j,4]>PS[i,4]:
                isMax=0
                break
        for j in range(i+1,N):
            if np.isnan(PS[i,5]) or np.isnan(PS[i,6]):
                continue
            if PS[j,1]>PS[i,1]+minDelta:
                break
            if np.linalg.norm(PS[j,2:3]-PS[i,2:3])>minD:
                continue
            if PS[j,4]>PS[i,4]:
                isMax=0
                break
        if isMax==1:
            L.append(int(PS[i,0]))
    return L



