import obspy
import numpy as np
from sacTool import getDataByFileName, staTimeMat
import os
from glob import glob
import matplotlib.pyplot as plt
import tool
import sacTool
from tool import Record, Quake, QuakeCC,getYmdHMSj
from multiprocessing import Process, Manager
import threading
import time
import math
from mathFunc import getDetec
import mapTool as mt
from distaz import DistAz

os.environ["MKL_NUM_THREADS"] = "10"
def predictLongData(model, x, N=2000, indexL=range(750, 1250)):
    if len(x) == 0:
        return np.zeros(0)
    N = x.shape[0]
    Y = np.zeros(N)
    perN = len(indexL)
    loopN = int(math.ceil(N/perN))
    perLoop = int(1000)
    inMat = np.zeros((perLoop, 2000, 1, 3))
    #print(len(x))
    for loop0 in range(0, int(loopN), int(perLoop)):
        loop1 = min(loop0+perLoop, loopN)
        for loop in range(loop0, loop1):
            i = loop*perN
            sIndex = min(max(0, i), N-2000)
            if sIndex > 0:
                inMat[loop-loop0, :, :, :] = processX(x[sIndex: sIndex+2000, :])\
                .reshape([2000, 1, 3])
        outMat = (model.predict(inMat)[:,:,:,:1]).reshape([-1, 2000])
        for loop in range(loop0, loop1):
            i = loop*perN
            sIndex = min(max(0, i), N-2000)
            if sIndex > 0:
                Y[indexL[0]+sIndex: indexL[-1]+1+sIndex] = \
                outMat[loop-loop0, indexL].reshape([-1])
    return Y


def processX(X, rmean=True, normlize=True, reshape=True):
    if reshape:
        X = X.reshape(-1, 2000, 1, 3)
    if rmean:
        X = X - X.mean(axis=(1, 2)).reshape([-1, 1, 1, 3])
    if normlize:
        X = X/(X.std(axis=(1, 2, 3)).reshape([-1, 1, 1, 1]))
    return X


def originFileName(net, station, comp, YmdHMSJ, dirL=['data/']):
    #dir='tmpSacFile/'
    sacFileNames = list()
    Y = YmdHMSJ
    for dir in dirL:
        sacFileNamesStr = dir+net+'.'+station+'.'+Y['Y']+Y['j']+\
            '*'+comp
        for file in glob(sacFileNamesStr):
            sacFileNames.append(file)
    #print(sacFileNames)
    return sacFileNames

class sta(object):
    def __init__(self, net, station, day, modelL=None, staTimeM=None,\
     loc=None, comp=['BHE','BHN','BHZ'], getFileName=originFileName, \
     freq=[-1, -1], mode='mid', isClearData=False,\
     taupM=tool.quickTaupModel(),isPre=True,delta0=0.02,R=[-91,91,\
    -181,181]):
        self.net = net
        self.loc = loc
        self.station = station
        self.day = day
        self.comp = comp
        self.taupM=taupM
        if loc[0]<R[0] or loc[0]>R[1] or loc[1]<R[2] or loc[1]>R[3]:
            self.data=sacTool.Data(np.zeros((0,3)))
            print('skip')
        else:
            self.data = getDataByFileName(self.getSacFileNamesL\
                (getFileName=getFileName), freq=freq,delta0=delta0,maxA=2e2)
        #print(len(sta.data.data))
        self.timeL = list()
        self.vL = list()
        self.mode = mode
        if isPre==True:
            indexLL = [range(750, 1250), range(1000, 1500)]
            if mode=='norm':
                minValueL=[0.5,0.5]
            if mode=='high':
                minValueL=[0.4, 0.4]
            if mode=='mid':
                minValueL=[0.25, 0.25]
            if mode=='low':
                minValueL=[0.2, 0.2]
            minDeltaL=[500, 750]
            for i in range(len(modelL)):
                tmpL = getDetec(predictLongData(modelL[i], self.data.data,\
                 indexL=indexLL[i]), minValue=minValueL[i], minDelta =\
                  minDeltaL[i])
                self.timeL.append(tmpL[0])
                self.vL.append(tmpL[1])
            self.pairD = self.getPSPair()
            self.isPick = np.zeros(len(self.pairD))
            self.orignM = self.convertPS2orignM(staTimeM)
            if isClearData:
                self.clearData()

    def __repr__(self):
        reprStr=self.net + ' '+self.station+\
        str(self.loc)
        return 'detec in station '+ reprStr


    def getSacFileNamesL(self, getFileName=originFileName):
        YmdHMSJ = getYmdHMSj(self.day)
        fNL=list();
        for comp in self.comp:
            fNL.append(getFileName(self.net, self.station,\
                comp, YmdHMSJ))
        return fNL

    def clearData(self):
        self.data.data = np.zeros((0, 3))

    def plotData(self):
        colorStr = ['.r', '.g']
        plt.plot(self.data.data[:,2]/self.data.data[:,2].max()\
            + np.array(0))
        for i in range(len(self.timeL)):
            plt.plot(self.timeL[i],self.vL[i], colorStr[i])
        plt.show()

    def calOrign(self, pTime, sTime):
        return self.taupM.get_orign_times(pTime, sTime, self.data.delta)

    def getPSPair(self, maxD=80):
        pairD = list()
        if len(self.timeL) == 0:
            return pairD
        if self.data.delta==0:
            return pairD
        maxN = maxD/self.data.delta
        pN=len(self.timeL[0])
        sN=len(self.timeL[1])
        j0=0
        for i in range(pN):
            pTime = self.timeL[0][i]
            if i < pN-1 and self.mode != 'low':
                pTimeNext = self.timeL[0][i+1]
            else:
                pTimeNext= self.timeL[0][i]+maxN
            pTimeNext = min(pTime+maxN, pTimeNext)
            isS = 0
            for j in range(j0, sN):
                if isS==0:
                    j0=j
                if self.timeL[1][j] > pTime and self.timeL[1][j] < pTimeNext:
                    sTime=self.timeL[1][j]
                    #print(pTime, sTime)
                    pairD.append([pTime*self.data.delta, sTime*self.data.delta\
                        , self.calOrign(pTime, sTime)*self.data.delta, \
                        (sTime-pTime)*self.data.delta, i, j])
                    isS=1
                if self.timeL[1][j] >= pTimeNext:
                    break
        return pairD

    def convertPS2orignM(self, staTimeM, maxDTime=2):
        laN = staTimeM.minTimeD.shape[0]
        loN = staTimeM.minTimeD.shape[1]
        orignM = [[list() for j in range(loN)] for i in range(laN)]
        if len(self.pairD)==0:
            return orignM
        bSec = self.data.bTime.timestamp
        timeL = np.zeros(len(self.pairD))
        for i in range(len(self.pairD)):
            timeL[i] = self.pairD[i][2]+bSec
        sortL = np.argsort(timeL)
        for i in sortL:
            for laIndex in range(laN):
                for loIndex in range(loN):
                    if self.pairD[i][3] >= staTimeM.minTimeD[laIndex][loIndex] - maxDTime \
                    and self.pairD[i][3] <= staTimeM.maxTimeD[laIndex][loIndex] + maxDTime:
                        pTime = self.pairD[i][0]+bSec
                        sTime = self.pairD[i][1]+bSec
                        timeTmp = [pTime, sTime, timeL[i], i]
                        orignM[laIndex][loIndex].append(timeTmp)
        return orignM

def argMax2D(M):
    maxValue = np.max(M)
    maxIndex = np.where(M==maxValue)
    return maxIndex[0][0], maxIndex[1][0]


def associateSta(staL, aMat, staTimeML, timeR=30, minSta=3, maxDTime=3, N=1, \
    isClearData=False, locator=None):
    timeN = int(timeR)*2
    startTime = obspy.UTCDateTime(2100, 1, 1)
    endTime = obspy.UTCDateTime(1970, 1, 1)
    staN = len(staL)
    for staIndex in range(staN):
        if isClearData:
            staL[staIndex].clearData()
        staL[staIndex].isPick = staL[staIndex].isPick*0
    for staTmp in staL:
        if len(staTmp.data.data) == 0:
            continue
        startTime = min(startTime, staTmp.data.bTime)
        endTime = max(endTime, staTmp.data.eTime)
    startSec = int(startTime.timestamp-90)
    endSec = int(endTime.timestamp+30)
    if N==1:
        quakeL=[]
        __associateSta(quakeL, staL, \
            aMat, staTimeML, startSec, \
            endSec, timeR=timeR, minSta=minSta, maxDTime=maxDTime,locator=locator)
        return quakeL
    for i in range(len(staL)):
        staL[i].clearData()
    manager=Manager()
    quakeLL=[manager.list() for i in range(N)]
    perN = int(int((endSec-startSec)/N+1)/timeN+1)*timeN
    processes=[]
    for i in range(N):
        process=Process(target=__associateSta, args=(quakeLL[i], \
            staL, aMat, staTimeML, startSec+i*perN, \
            startSec+(i+1)*perN+1))
        #process.setDaemon(True)
        process.start()
        processes.append(process)

    for process in processes:
        print(process)
        process.join()
    quakeL=list()

    for quakeLTmp in quakeLL:
        for quakeTmp in quakeLTmp:
            quakeL.append(quakeTmp)
    return quakeL
    

def __associateSta(quakeL, staL, aMat, staTimeML, startSec, endSec, \
    timeR=30, minSta=3, maxDTime=3, locator=None):
    print('start', startSec, endSec)
    laN = aMat.laN
    loN = aMat.loN
    staN = len(staL)
    timeN = int(timeR)*10
    stackM = np.zeros((timeN*3, laN, loN))
    tmpStackM=np.zeros((timeN*3+3*maxDTime, laN, loN))
    stackL = np.zeros(timeN*3)
    staMinTimeL=np.ones(staN)*0
    quakeCount=0

    for loop in range(2):
        staOrignMIndex = np.zeros((staN, laN, loN), dtype=int)
        staMinTimeL=np.ones(staN)*0
        count=0
        for sec0 in range(startSec, endSec, timeN):
            count=count+1
            if count%10==0:
                print('process:',(sec0-startSec)/(endSec-startSec)*100,'%  find:',len(quakeL))
            stackM[0:2*timeN, :, :] = stackM[timeN:, :, :]
            stackM[2*timeN:, :, :] = stackM[0:timeN, :, :]*0
            tmpStackM=tmpStackM*0
            st=sec0+2*timeN - maxDTime
            et=sec0+3*timeN + maxDTime
            for staIndex in range(staN):
                tmpStackM=tmpStackM*0
                for laIndex in range(laN):
                    for loIndex in range(loN):
                        if len(staL[staIndex].orignM[laIndex][loIndex])>0:
                            index0=staOrignMIndex[staIndex, laIndex, loIndex]
                            for index in range(index0, len(staL[staIndex].orignM[laIndex][loIndex])):
                                timeT = staL[staIndex].orignM[laIndex][loIndex][index][2]
                                pairIndex = staL[staIndex].orignM[laIndex][loIndex][index][3]
                                if timeT >et:
                                    staOrignMIndex[staIndex, laIndex, loIndex] = index
                                    break
                                if timeT > st and staL[staIndex].isPick[pairIndex]==0:
                                    pIndex = staL[staIndex].pairD[pairIndex][4]
                                    sIndex = staL[staIndex].pairD[pairIndex][5]
                                    pTime = staL[staIndex].timeL[0][pIndex]
                                    sTime = staL[staIndex].timeL[1][sIndex]
                                    staOrignMIndex[staIndex, laIndex, loIndex] = index
                                    if pTime * sTime ==0:
                                        continue
                                    #if laIndex==30 and loIndex==10:
                                    #    print(staIndex,laIndex,loIndex,timeT)
                                    for dt in range(-maxDTime, maxDTime+1):
                                        tmpStackM[int(timeT-sec0+dt), laIndex, loIndex]=1
                stackM[2*timeN: 3*timeN, :, :] += tmpStackM[2*timeN: 3*timeN, :, :]

            stackL = stackM.max(axis=(1,2))
            peakL, peakN = tool.getDetec(stackL, minValue=minSta, minDelta=timeR)

            for peak in peakL:
                if peak > timeN and peak <= 2*timeN:
                    time = peak + sec0
                    laIndex, loIndex = argMax2D(stackM[peak, :, :].reshape((laN, loN)))
                    quakeCount+=1
                    quake = Quake(loc=[aMat[laIndex][loIndex].midLa, aMat[laIndex][loIndex].midLo,10.0],\
                        time=time, randID=quakeCount)
                    for staIndex in range(staN):
                        isfind=0
                        if len(staL[staIndex].orignM[laIndex][loIndex]) != 0:
                            for index in range(staOrignMIndex[staIndex, laIndex, loIndex], -1, -1):
                                if int(abs(staL[staIndex].orignM[laIndex][loIndex][index][2]-time))<=maxDTime:
                                    if staL[staIndex].isPick[staL[staIndex].\
                                            orignM[laIndex][loIndex][index][3]]==0:
                                        pairDIndex = staL[staIndex].orignM[laIndex][loIndex][index][3]
                                        pIndex = staL[staIndex].pairD[pairDIndex][4]
                                        sIndex = staL[staIndex].pairD[pairDIndex][5]
                                        if staL[staIndex].timeL[0][pIndex] > 0 and \
                                                staL[staIndex].timeL[1][sIndex] > 0:
                                            quake.append(Record(staIndex, \
                                                staL[staIndex].orignM[laIndex][loIndex][index][0], \
                                                staL[staIndex].orignM[laIndex][loIndex][index][1]))
                                            isfind=1
                                            staL[staIndex].timeL[0][pIndex] = 0
                                            staL[staIndex].timeL[1][sIndex] = 0
                                            staL[staIndex].isPick[pairDIndex] = 1
                                            break
                                if staL[staIndex].orignM[laIndex][loIndex][index][2] < time - maxDTime:
                                    break
                            if isfind==0:
                                pTime=0
                                sTime=0
                                pTimeL=staL[staIndex].timeL[0]*staL[staIndex].data.delta\
                                +staL[staIndex].data.bTime.timestamp
                                sTimeL=staL[staIndex].timeL[1]*staL[staIndex].data.delta\
                                +staL[staIndex].data.bTime.timestamp
                                pTimeMin=time+staTimeML[staIndex].minTimeP[laIndex,loIndex]-maxDTime
                                pTimeMax=time+staTimeML[staIndex].maxTimeP[laIndex,loIndex]+maxDTime
                                sTimeMin=time+staTimeML[staIndex].minTimeS[laIndex,loIndex]-maxDTime
                                sTimeMax=time+staTimeML[staIndex].maxTimeS[laIndex,loIndex]+maxDTime
                                validP=np.where((pTimeL/1e5-pTimeMin/1e5)*(pTimeL/1e5-pTimeMax/1e5)<=0)
                                if len(validP)>0:
                                    if len(validP[0])>0:
                                        pTime=pTimeL[validP[0]][0]
                                        pIndex=validP[0][0]
                                if pTime < 1:
                                    continue
                                validS=np.where((sTimeL-sTimeMin)*(sTimeL-sTimeMax) < 0)
                                if len(validS)>0:
                                    if len(validS[0])>0:
                                        sTime=sTimeL[validS[0]][0]
                                        sIndex=validS[0][0]
                                if pTime > 1:
                                    if sTime <1  and staL[staIndex].vL[0][pIndex]<0.3:
                                        continue
                                    staL[staIndex].timeL[0][pIndex]=0
                                    if sTime >1:
                                        staL[staIndex].timeL[1][sIndex]=0
                                    quake.append(Record(staIndex, pTime, sTime))
                    if locator != None and len(quake)>=3:
                        try:
                            quake,res=locator.locate(quake,maxDT=50)
                            print(quake.time,quake.loc,res)
                        except:
                            print('wrong in locate')
                        else:
                            pass
                    quakeL.append(quake)
    return quakeL

def getStaTimeL(staInfos, aMat,taupM=tool.quickTaupModel()):
    #manager=Manager()
    #staTimeML=manager.list()
    staTimeML=list()
    for staInfo in staInfos:
        loc=[staInfo['la'],staInfo['lo']]
        staTimeML.append(staTimeMat(loc, aMat, taupM=taupM))
    return staTimeML

def getSta(staL,i, nt, st, date, modelL, staTimeM, loc, \
        freq,getFileName,taupM, mode,isPre=True,R=[-90,90,\
    -180,180]):
    staL[i] = sta(nt, st, date, modelL, staTimeM, loc, \
            freq=freq, getFileName=getFileName, taupM=taupM, mode=mode,isPre=isPre,R=R)


def getStaL(staInfos, aMat, staTimeML, modelL, date, getFileName=originFileName,\
    taupM=tool.quickTaupModel(), mode='mid', N=5,isPre=True,f=[2, 15],R=[-90,90,\
    -180,180]):
    staL=[None for i in range(len(staInfos))]
    threads = list()
    for i in range(len(staInfos)):
        staInfo=staInfos[i]
        nt = staInfo['net']
        st = staInfo['sta']
        loc = [staInfo['la'],staInfo['lo']]
        print('process on sta: ',i)
        getSta(staL, i, nt, st, date, modelL, staTimeML[i], loc, \
            f, getFileName, taupM, mode,isPre=isPre,R=R)
    return staL
    for i in range(len(threads)):
        print('process on sta: ',i)
        thread = threads[i]
        while threading.activeCount()>N:
            time.sleep(0.1)
        thread.start()

    for i in range(len(threads)):
        threads[i].join()
        print('sta: ',i,' completed')
         
    return staL

def plotQuakeDis(quakeLs,output='quakeDis.png',cmd='.b',markersize=0.8,\
    alpha=0.3,R=None,topo=None,m=None,staInfos=None,minSta=8,minCover=0.8,\
    faultFile="Chinafault_fromcjw.dat",mul=1,loL0=[],laL0=[],isBox=False):
    la=[]
    lo=[]
    dep=[]
    mlL=[]
    plt.close()
    plt.figure(figsize=[12,8])
    for quakeL in quakeLs:
        for quake in quakeL:
            if len(quake)<minSta:
                continue
            if staInfos!=None:
                if quake.calCover(staInfos)<minCover:
                    continue
            ml=0
            if quake.ml !=None:
                if quake.ml>-2:
                    ml=quake.ml
            la.append(quake.loc[0])
            lo.append(quake.loc[1])
            dep.append(quake.loc[2])
            mlL.append(ml)
    la=np.array(la)
    lo=np.array(lo)
    dep=np.array(dep)
    mlL=np.array(mlL)
    if R==None:
        R=[la.min(),la.max(),lo.min(),lo.max()]
    #print(R)
    
    if m==None:
        m=mt.genBaseMap(R=R,topo=topo)
    if not staInfos == None:
        sla=[]
        slo=[]
        sdep=[]
        for staInfo in staInfos:
            sla.append(staInfo['la'])
            slo.append(staInfo['lo'])
            sdep.append(staInfo['dep'])
        sla=np.array(sla)
        slo=np.array(slo)
        sdep=np.array(sdep)
        hS,=mt.plotOnMap(m,sla,slo,'^r',markersize=5,alpha=1,linewidth=1)
    faultL=mt.readFault(faultFile)
    hF=None
    for fault in faultL:
        if fault.inR(R):
            hFTmp,=fault.plot(m)
            if hFTmp!=None:
                hF=hFTmp
    if len(laL0)>1:
        hC,=mt.plotOnMap(m,laL0,loL0,'ok',markersize=2,mfc=[1,1,1])
    if isBox:
        laLB0=[38.7,42.2]
        loLB0=[97.5,103.8]
        laLB=np.array([laLB0[0],laLB0[0],laLB0[1],laLB0[1],laLB0[0]])
        loLB=np.array([loLB0[1],loLB0[0],loLB0[0],loLB0[1],loLB0[1]])
        mt.plotOnMap(m,laLB,loLB,'r',linewidth=3,markersize=3)
    hQ,=mt.plotOnMap(m,la,lo,cmd,markersize,alpha)
    #mt.scatterOnMap(m,la,lo,s=np.exp(mlL/1.5)*mul,alpha=alpha,c=np.array([1,0,0]))
    parallels = np.arange(0.,90,2.)

    if len(laL0)>1:
        #hC,=mt.plotOnMap(m,laL0,loL0,'ok',markersize=2)
        plt.legend((hQ,hC,hS,hF),('Quakes','Catalog','Station','Faults'),\
            bbox_to_anchor=(1, 1),loc='lower right')
    else:
        plt.legend((hQ,hS,hF),('Quakes','Station','Faults'),bbox_to_anchor=(1, 1),\
              loc='lower right')
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(10.,360.,2.)
    plt.gca().yaxis.set_ticks_position('left')
    m.drawmeridians(meridians,labels=[True,False,False,True])
    plt.savefig(output)
    return m

def getStaLByQuake(staInfos, aMat, staTimeML, modelL,quake,\
    getFileName=originFileName,taupM=tool.quickTaupModel(), \
    mode='mid', N=5,isPre=False,bTime=-100,delta0=0.02):
    staL=[None for i in range(len(staInfos))]
    threads = list()
    for i in range(len(staInfos)):
        staInfo=staInfos[i]
        nt = staInfo['net']
        st = staInfo['sta']
        loc = [staInfo['la'],staInfo['lo']]
        print('process on sta: ',i)
        dis=DistAz(quake.loc[0],quake.loc[1],staInfos[i]['la'],\
            staInfos[i]['lo']).getDelta()
        date=obspy.UTCDateTime(quake.time+taupM.get_travel_times(quake.loc[2],dis)[0].time+bTime)
        getSta(staL, i, nt, st, date, modelL, staTimeML[i], loc, \
            [0.01, 15], getFileName, taupM, mode,isPre=isPre,delta0=delta0)
    return staL


##plot part
def plotRes(staL, quake, filename=None):
    colorStr='br'
    for record in quake:
        color=0
        pTime=record[1]
        sTime=record[2]
        staIndex=record[0]
        if staIndex>100:
            color=1
        print(staIndex,pTime, sTime)
        st=quake.time-10
        et=sTime+10
        if sTime==0:
            et=pTime+30
        pD=(pTime-quake.time)%1000
        if pTime ==0:
            pD = ((sTime-quake.time)/1.73)%1000
        if staL[staIndex].data.data.size<100:
            continue
        print(st, et, staL[staIndex].data.delta)
        timeL=np.arange(st, et, staL[staIndex].data.delta)
        #data = staL[staIndex].data.getDataByTimeL(timeL)
        data=staL[staIndex].data.getDataByTimeLQuick(timeL)
        if timeL.shape[0] != data.shape[0]:
            print('not same length for plot')
            continue
        if timeL.size<1:
            print("no timeL for plot")
            continue
        plt.plot(timeL, data[:, 2]/data[:,2].max()+pD,colorStr[color])
        plt.text(timeL[0],pD+0.5,staL[staIndex].station)
        if pTime>0:
            plt.plot([pTime, pTime], [pD+2, pD-2], 'g')
            if isinstance(quake,QuakeCC):
                plt.text(pTime+1,pD+0.5,'%.2f'%record.getPCC())
        if sTime >0:
            plt.plot([sTime, sTime], [pD+2, pD-2], 'k')
            if isinstance(quake,QuakeCC):
                plt.text(sTime+1,pD+0.5,'%.2f'%record.getSCC())
    if isinstance(quake,QuakeCC):
        plt.title('%s %.3f %.3f %.3f cc:%.3f' % (obspy.UTCDateTime(quake.time).\
            ctime(), quake.loc[0], quake.loc[1],quake.loc[2],quake.cc))
    else:
        plt.title('%s %.3f %.3f %.3f' % (obspy.UTCDateTime(quake.time).\
            ctime(), quake.loc[0], quake.loc[1],quake.loc[2]))
    if filename==None:
        plt.show()
    if filename!=None:
        dayDir=os.path.dirname(filename)
        if not os.path.exists(dayDir):
            os.mkdir(dayDir)
        plt.savefig(filename)
        plt.close()

def plotResS(staL,quakeL, outDir='output/'):
    for quake in quakeL:
        filename=outDir+'/'+quake.filename[0:-3]+'png'
        #filename=outDir+'/'+str(quake.time)+'.jpg'
        plotRes(staL,quake,filename=filename)

def plotQuakeCCDis(quakeCCLs,quakeRefL,output='quakeDis.png',cmd='.r',markersize=0.8,\
    alpha=0.3,R=None,topo=None,m=None,staInfos=None,minSta=8,minCover=0.8,\
    faultFile="Chinafault_fromcjw.dat",mul=1,minCC=0.5):
    la=[]
    lo=[]
    dep=[]
    mlL=[]
    count=0
    for quakeL in quakeCCLs:
        for quake in quakeL:
            if len(quake)<minSta:
                continue
            if staInfos!=None:
                if quake.calCover(staInfos,minCC=minCC)<minCover:
                    continue
            ml=0
            if quake.ml !=None:
                if quake.ml>-2:
                    ml=quake.ml
            la.append(quake.loc[0])
            lo.append(quake.loc[1])
            dep.append(quake.loc[2])
            mlL.append(ml)
            count+=1
    print(count)
    la=np.array(la)
    lo=np.array(lo)
    dep=np.array(dep)
    mlL=np.array(mlL)
    if R==None:
        R=[la.min(),la.max(),lo.min(),lo.max()]
    laR=[]
    loR=[]
    depR=[]
    mlLR=[]
    for quake in quakeRefL:
        ml=0
        if quake.ml !=None:
            if quake.ml>-2:
                ml=quake.ml
        laR.append(quake.loc[0])
        loR.append(quake.loc[1])
        depR.append(quake.loc[2])
        mlLR.append(ml)
    laR=np.array(laR)
    loR=np.array(loR)
    depR=np.array(depR)
    mlLR=np.array(mlLR)
    if m==None:
        m=mt.genBaseMap(R=R,topo=topo)
    if not staInfos == None:
        sla=[]
        slo=[]
        sdep=[]
        for staInfo in staInfos:
            sla.append(staInfo['la'])
            slo.append(staInfo['lo'])
            sdep.append(staInfo['dep'])
        sla=np.array(sla)
        slo=np.array(slo)
        sdep=np.array(sdep)
        hS,=mt.plotOnMap(m,sla,slo,'^k',markersize=5,alpha=1)
    faultL=mt.readFault(faultFile)
    hF=None
    for fault in faultL:
        if fault.inR(R):
            hFTmp,=fault.plot(m)
            if hFTmp!=None:
                hF=hFTmp

    hT,=mt.plotOnMap(m,laR,loR,'*b',markersize*2,1)
    hCC,=mt.plotOnMap(m,la,lo,cmd,markersize,alpha)
    print(len(laR),len(la))

    plt.legend((hT,hCC,hS,hF),('Templates','Microearthquakes','Station','Faults'))
    #mt.plotOnMap(m,laR,loR,'*k',markersize*2,1)
    #mt.scatterOnMap(m,la,lo,s=np.exp(mlL/1.5)*mul,alpha=alpha,c=np.array([1,0,0]))
    plt.title('minSta:%d minCover:%.1f minCC:%.1f MFT:%d'%(minSta,minCover,minCC,count))
    dD=max(int((R[1]-R[0])*10)/40,int((R[3]-R[2])*10)/40)
    parallels = np.arange(int(R[0]),int(R[1]+1),dD)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(int(R[2]),int(R[3]+1),dD)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    return m


def showExample(filenameL,modelL,delta=0.02,t=[]):
    data=getDataByFileName(filenameL,freq=[2,15])
    data=data.data[:2000*50]
    
    #i0=int(750/delta)
    #i1=int(870/delta)
    #plt.specgram(np.sign(data[i0:i1,1])*(np.abs(data[i0:i1,1])**0.5),NFFT=200,Fs=50,noverlap=190)
    data/=data.max()/2
    #plt.colorbar()
    #plt.show()
    plt.close()
    yL=[predictLongData(modelL[i],data) for i in range(2)]
    timeL=np.arange(data.shape[0])*delta-720
    #print(data.shape,timeL.shape)
    for i in range(3):
        plt.plot(timeL,np.sign(data[:,i])*(np.abs(data[:,i]))+i,'k',linewidth=0.3)
    for i in range(2):
        plt.plot(timeL,yL[i]-i-1.5,'k',linewidth=0.5)
    if len(t)>0:
        plt.xlim(t)
    plt.yticks(np.arange(-2,3),['S','P','E','N','Z'])
    plt.ylim([-2.7,3])
    plt.xlabel('t/s')
    plt.savefig('NM/complexCondition.eps')
    plt.savefig('NM/complexCondition.tiff',dpi=300)
    plt.close()
    

def showExampleV2(filenameL,modelL,delta=0.02,t=[],staName='sta'):
    data=getDataByFileName(filenameL,freq=[2,15])
    data=data.data[:3500*50]
    
    #i0=int(750/delta)
    #i1=int(870/delta)
    #plt.specgram(np.sign(data[i0:i1,1])*(np.abs(data[i0:i1,1])**0.5),NFFT=200,Fs=50,noverlap=190)
    data/=data.max()/2
    #plt.colorbar()
    #plt.show()
    plt.close()
    yL=[predictLongData(model,data) for model in modelL]
    timeL=np.arange(data.shape[0])*delta-720
    #print(data.shape,timeL.shape)
    for i in range(3):
        plt.plot(timeL,np.sign(data[:,i])*(np.abs(data[:,i]))+i,'k',linewidth=0.3)
    for i in range(len(modelL)):
        plt.plot(timeL,yL[i]-i-1.5,'k',linewidth=0.5)
        #plt.plot(timeL,yL[i]*0+0.5-i-1.5,'--k',linewidth=0.5)
    if len(t)>0:
        plt.xlim(t)
    plt.yticks(np.arange(-4,3),['S1','S0','P1','P0','E','N','Z'])
    plt.ylim([-4.7,3])
    plt.xlabel('t/s')
    plt.savefig('NM/complexConditionV2_%s.eps'%staName)
    plt.savefig('NM/complexConditionV2_%s.tiff'%staName,dpi=300)
    plt.close()
