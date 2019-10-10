import numpy as np
from distaz import DistAz
from obspy import UTCDateTime
from tool import getYmdHMSj
from numba import jit
import matplotlib.pyplot as plt
import os
from mathFunc import xcorr

def preEvent(quakeL,staInfos,filename='TOMODD/input/event.dat'):
    with open(filename,'w+') as f:
        for i in range(len(quakeL)):
            quake=quakeL[i]
            ml=0
            if quake.ml!=None :
                if quake.ml>-2:
                    ml=quake.ml
            Y=getYmdHMSj(UTCDateTime(quake.time))
            f.write("%s  %s%02d   %.4f   %.4f    % 7.3f % 5.2f   0.15    0.51  % 5.2f   % 8d %1d\n"%\
                (Y['Y']+Y['m']+Y['d'],Y['H']+Y['M']+Y['S'],int(quake.time*100)%100,\
                    quake.loc[0],quake.loc[1],max(5,quake.loc[2]),ml,1,i,0))

def preABS(quakeL,staInfos,filename='TOMODD/input/ABS.dat'):
    with open(filename,'w+') as f:
        for i in range(len(quakeL)):
            quake=quakeL[i]
            f.write('#  % 8d\n'%i)
            for record in quake:
                staIndex=record.getStaIndex()
                staInfo=staInfos[staIndex]
                if record.pTime()!=0:
                    f.write('%s     %7.2f   %5.3f   P\n'%(staInfo['nickName'],record.pTime()-quake.time,1.0))
                if record.sTime()!=0:
                    f.write('%s     %7.2f   %5.3f   S\n'%(staInfo['nickName'],record.sTime()-quake.time,1.0))

def preSta(staInfos,filename='TOMODD/input/station.dat'):
    with open(filename,'w+') as f:
        for staInfo in staInfos:
            f.write('%s %7.4f %8.4f %.0f\n'%(staInfo['nickName'],staInfo['la'],staInfo['lo'],staInfo['dep']))

def sameSta(timeL1,timeL2):
    return np.where(np.sign(timeL1*timeL2)>0)[0]

def calDT(quake0,quake1,waveform0,waveform1,staInfos,bSec0=-2,eSec0=3,\
    bSec1=-3,eSec1=4,delta=0.02,minC=0.6,maxD=0.3,minSameSta=5):
    pTime0=quake0.getPTimeL(staInfos)
    sTime0=quake0.getSTimeL(staInfos)
    pTime1=quake1.getPTimeL(staInfos)
    sTime1=quake1.getSTimeL(staInfos)
    sameIndex=sameSta(pTime0,pTime1)
    if len(sameIndex)<minSameSta:
        return None
    if DistAz(quake0.loc[0],quake0.loc[1],quake1.loc[0],quake1.loc[1]).getDelta()>maxD:
        return None
    dT=[];
    timeL0=np.arange(bSec0,eSec0,delta)
    indexL0=(timeL0/delta).astype(np.int64)-waveform0['indexL'][0][0]
    timeL1=np.arange(bSec1,eSec1,delta)
    indexL1=(timeL1/delta).astype(np.int64)-waveform1['indexL'][0][0]
    for staIndex in sameIndex:
        if pTime0[staIndex]!=0 and pTime1[staIndex]!=0:
            index0=np.where(waveform0['staIndexL'][0]==staIndex)[0]
            pWave0=waveform0['pWaveform'][index0,indexL0,2]
            index1=np.where(waveform1['staIndexL'][0]==staIndex)[0]
            #print(index1)
            pWave1=waveform1['pWaveform'][index1,indexL1,2]
            c=xcorr(pWave1,pWave0)
            maxC=c.max()
            if maxC>minC:
                maxIndex=c.argmax()
                dt=timeL1[maxIndex]-timeL0[0]
                dT.append([dt,maxC,staIndex,1])
        if sTime0[staIndex]!=0 and sTime1[staIndex]!=0:
            index0=np.where(waveform0['staIndexL'][0]==staIndex)[0]
            sWave0=waveform0['sWaveform'][index0,indexL0,0]
            index1=np.where(waveform1['staIndexL'][0]==staIndex)[0]
            sWave1=waveform1['sWaveform'][index1,indexL1,0]
            c=xcorr(sWave1,sWave0)
            maxC0=c.max()
            if maxC0>minC:
                maxIndex=c.argmax()
                dt=timeL1[maxIndex]-timeL0[0]
                dT.append([dt,maxC0,staIndex,2])
            index0=np.where(waveform0['staIndexL'][0]==staIndex)[0]
            sWave0=waveform0['sWaveform'][index0,indexL0,1]
            index1=np.where(waveform1['staIndexL'][0]==staIndex)[0]
            sWave1=waveform1['sWaveform'][index1,indexL1,1]
            c=xcorr(sWave1,sWave0)
            maxC1=c.max()
            if maxC1>minC and maxC1>maxC0:
                maxIndex=c.argmax()
                dt=timeL1[maxIndex]-timeL0[0]
                dT.append([dt,maxC1,staIndex,2])
    return dT

def calDTM(quakeL,waveformL,staInfos,maxD=0.3,minC=0.6,minSameSta=5):
    dTM=[[None for quake in quakeL]for quake in quakeL]
    for i in range(len(quakeL)):
        print(i)
        for j in range(i+1,len(quakeL)):
            dTM[i][j]=calDT(quakeL[i],quakeL[j],waveformL[i],waveformL[j],\
                staInfos,maxD=maxD,minC=minC,minSameSta=minSameSta)
    return dTM

def plotDT(waveformL,dTM,i,j,staInfos,bSec0=-2,eSec0=3,\
    bSec1=-3,eSec1=4,delta=0.02,minSameSta=5):
    waveform0=waveformL[i]
    waveform1=waveformL[j]
    timeL0=np.arange(bSec0,eSec0,delta)
    indexL0=(timeL0/delta).astype(np.int64)-waveform0['indexL'][0][0]
    timeL1=np.arange(bSec1,eSec1,delta)
    indexL1=(timeL1/delta).astype(np.int64)-waveform1['indexL'][0][0]
    count=0
    staIndexL0=waveform0['staIndexL'][0].astype(np.int64)
    staIndexL1=waveform1['staIndexL'][0].astype(np.int64)
    for dT in dTM[i][j]:
        staIndex=dT[2]
        tmpIndex0=np.where(staIndexL0==staIndex)[0][0]
        tmpIndex1=np.where(staIndexL1==staIndex)[0][0]
        print(tmpIndex0,tmpIndex1)
        if dT[3]==1:
            w0=waveform0['pWaveform'][int(tmpIndex0),indexL0,2]
            w1=waveform1['pWaveform'][int(tmpIndex1),indexL1,2]
        else:
            continue
            w0=waveform0['sWaveform'][int(tmpIndex0),indexL0,0]
            w1=waveform1['sWaveform'][int(tmpIndex1),indexL1,0]
        plt.plot(timeL0+dT[0],w0/(w0.max())*0.5+count,'r')
        print(xcorr(w1,w0).max())
        plt.plot(timeL1,w1/(w1.max())*0.5+count,'b')
        plt.plot(timeL0-dT[0],w0/(w0.max())*0.5+count+2,'r')
        #print(w0.max())
        plt.plot(timeL1,w1/(w1.max())*0.5+count+2,'b')
        #plt.plot(+count,'g')
        #print((w1/w1.max()).shape)
        plt.text(timeL1[0],count+0.5,'cc=%.2f dt=%.2f maxCC:'%(dT[1],dT[0]))
        count+=1
        plt.show()

def saveDTM(dTM,filename):
    N=len(dTM)
    with open(filename,'w+') as f:
        f.write("# %d\n"%N)
        for i in range(N):
            for j in range(N):
                if dTM[i][j]==None:
                    continue
                f.write("i %d %d\n"%(i,j))
                for dt in dTM[i][j]:
                    f.write("%f %f %d %d\n"%(dt[0],dt[1],dt[2],dt[3]))
def loadDTM(filename='dTM'):
    with open(filename) as f:
        for line in f.readlines():
            if line.split()[0]=='#':
                N=int(line.split()[1])
                dTM=[[None for i in range(N)]for i in range(N)]
                continue
            if line.split()[0]=='i':
                i=int(line.split()[1])
                j=int(line.split()[2])
                dTM[i][j]=[]
                continue
            staIndex=int(line.split()[2])
            dTM[i][j].append([float(line.split()[0]),float(line.split()[1]),\
            int(line.split()[2]),int(line.split()[3])])
    return dTM

def reportDTM(dTM):
    N=len(dTM)
    sumN=np.zeros(N)
    quakeN=np.zeros(N)
    for i in range(N):
        for j in range(N):
            if dTM[i][j]!=None:
                if len(dTM)<=0:
                    continue
                sumN[i]+=len(dTM[i][j])
                sumN[j]+=len(dTM[i][j])
                quakeN[i]+=1
                quakeN[j]+=1
    plt.subplot(2,1,1)
    plt.plot(sumN)
    plt.subplot(2,1,2)
    plt.plot(quakeN)
    plt.show()


def preDTCC(quakeL,staInfos,dTM,maxD=0.5,minSameSta=5,minPCC=0.75,minSCC=0.75,perCount=500,\
    filename='TOMODD/input/dt.cc'):
    N=len(quakeL)
    with open(filename,'w+') as f:
        for i in range(len(quakeL)):
            print(i)
            pTime0=quakeL[i].getPTimeL(staInfos)
            sTime0=quakeL[i].getSTimeL(staInfos)
            time0=quakeL[i].time
            count=0
            for j in range(i+1,len(quakeL)):
                if dTM[i][j]==None:
                    continue
                if count>perCount*(1-i/N):
                    break
                if DistAz(quakeL[j].loc[0],quakeL[j].loc[1],quakeL[j].loc[0],quakeL[j].loc[1]).getDelta()>maxD:
                    continue
                pTime1=quakeL[j].getPTimeL(staInfos)
                sTime1=quakeL[j].getSTimeL(staInfos)
                time1=quakeL[j].time
                if len(sameSta(pTime0,pTime1))<minSameSta:
                    continue                  
                for dtD in dTM[i][j]:
                    dt,maxC,staIndex,phaseType=dtD
                    if phaseType==1 and maxC>minPCC:
                        dt=pTime0[staIndex]-time0-(pTime1[staIndex]-time1+dt)
                        f.write("% 9d % 9d %s %8.3f %6.4f %s\n"%(i,j,staInfos[staIndex]['nickName'],dt,maxC*maxC,'P'))
                        coutn=count+1
                    if phaseType==2 and maxC>minSCC:
                        dt=sTime0[staIndex]-time0-(sTime1[staIndex]-time1+dt)
                        f.write("% 9d % 9d %s %8.3f %6.4f %s\n"%(i,j,staInfos[staIndex]['nickName'],dt,maxC*maxC,'S'))
                        coutn=count+1


def preMod(R,nx=8,ny=8,nz=8,filename='TOMODD/MOD'):
    with open(filename,'w+') as f:
        vp=[5.5,5.90,  6.01,  6.56, 6.91, 8.40, 8.79, 8.88, 9.00]
        vs=[2.4,2.67, 3.01,  4.10, 4.24, 4.50, 5.00, 5.15, 6.00]
        x=np.zeros(nx)
        y=np.zeros(ny)
        z=[-150,  7.5,  15,  25, 37,  53, 90, 150,  500]
        f.write('0.1 %d %d %d\n'%(nx,ny,nz))
        x[0]=R[2]-5
        x[-1]=R[3]+5
        y[0]=R[0]-5
        y[-1]=R[1]+5
        x[1]=(x[0]+R[2])/2
        x[-2]=(x[-1]+R[3])/2
        y[1]=(y[0]+R[0])/2
        y[-2]=(y[-1]+R[1])/2
        x[2:-2]=np.arange(R[2],R[3]+0.001,(R[3]-R[2])/(nx-5))
        y[2:-2]=np.arange(R[0],R[1]+0.001,(R[1]-R[0])/(ny-5))
        #f.write("\n")
        for i in range(nx):
            f.write('%.4f '%x[i])
        f.write('\n')
        for i in range(ny):
            f.write('%.4f '%y[i])
        f.write('\n')
        for i in range(nz):
            f.write('%.4f '%z[i])
        f.write('\n')

        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    f.write('%.2f '%vp[i])
                f.write('\n')

        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    f.write('%.2f '%(vp[i]/vs[i]))
                f.write('\n')

def getReloc(quakeL,filename='TOMODD/tomoDD.reloc'):
    quakeRelocL=[]
    with open(filename) as f:
        for line in f.readlines():
            line=line.split()
            time=quakeL[0].tomoTime(line)
            index=int(line[0])
            print(quakeL[index].time-time)
            quakeRelocL.append(quakeL[index])
            quakeRelocL[-1].getReloc(line)
    return quakeRelocL

