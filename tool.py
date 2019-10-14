import numpy as np
import h5py
import scipy.io as sio
from scipy import interpolate as interp
from obspy import UTCDateTime, taup
import obspy
from multiprocessing import Process, Manager
from mathFunc import matTime2UTC,rad2deg
import os
import re
from glob import glob
from distaz import DistAz
import matplotlib.pyplot as plt
from scipy import signal
from openpyxl import Workbook


def getYmdHMSj(date):
    YmdHMSj = {}
    YmdHMSj.update({'Y': date.strftime('%Y')})
    YmdHMSj.update({'m': date.strftime('%m')})
    YmdHMSj.update({'d': date.strftime('%d')})
    YmdHMSj.update({'H': date.strftime('%H')})
    YmdHMSj.update({'M': date.strftime('%M')})
    YmdHMSj.update({'S': date.strftime('%S')})
    YmdHMSj.update({'j': date.strftime('%j')})
    YmdHMSj.update({'J': date.strftime('%j')})
    return YmdHMSj


class Record(list):
    '''
    the basic class 'Record' used for single station's P and S record
    it's based on python's default list
    [staIndex pTime sTime]
    we can setByLine
    we provide a way to copy
    '''
    def __init__(self,staIndex=-1, pTime=-1, sTime=-1):
        super(Record, self).__init__()
        self.append(staIndex)
        self.append(pTime)
        self.append(sTime)
    def __repr__(self):
        return self.summary()
    def summary(self):
        Summary='%d'%self[0]
        for i in range(1,len(self)):
            Summary=Summary+" %f "%self[i]
        Summary=Summary+'\n'
        return Summary
    def copy(self):
        return Record(self[0],self[1],self[2])
    def getStaIndex(self):
        return self[0]
    def staIndex(self):
        return self.getStaIndex()
    def pTime(self):
        return self[1]
    def sTime(self):
        return self[2]
    def setByLine(self,line):
        if isinstance(line,str):
            line=line.split()
        self[0]=int(line[0])
        for i in range(1,len(self)):
            self[i]=float(line[i])
        return self
    def set(self,line,mode='byLine'):
        return self.setByLine(line)

#MLI  1976/01/01 01:29:39.6 -28.61 -177.64  59.0 6.2 0.0 KERMADEC ISLANDS REGION 
class Quake(list):
    '''
    a basic class 'Quake' used for quake's information and station records
    it's baed on list class
    the elements in list are composed by Records
    the basic information: loc, time, ml, filename, randID
    we use randID to distinguish time close quakes
    you can set the filename storting the waveform by yourself in .mat form
    otherwise, we would generate the file like dayNum/time_randID.mat

    we provide way for set by line/WLX/Mat and from IRIS/NDK
    '''
    def __init__(self, loc=[-999, -999,10], time=-1, randID=None, filename=None, ml=None):
        super(Quake, self).__init__()
        self.loc = [0,0,0]
        self.loc[:len(loc)]=loc
        self.time = time
        self.ml=ml
        if randID != None:
            self.randID=randID
        else:
            self .randID=int(10000*np.random.rand(1))
        if filename != None:
            self.filename = filename
        else:
            self.filename = self.getFilename()
    def __repr__(self):
        return self.summary()
    def summary(self,count=0):
        ml=-9
        if self.ml!=None:
            ml=self.ml
        Summary='quake: %f %f %f num: %d index: %d randID: %d filename: \
            %s %f %f\n' % (self.loc[0], self.loc[1],\
            self.time, len(self),count, self.randID, \
            self.filename,ml,self.loc[2])
        return Summary

    def getFilename(self):
        dayDir = str(int(self.time/86400))+'/'
        return dayDir+str(int(self.time))+'_'+str(self.randID)+'.mat'

    def append(self, addOne):
        if isinstance(addOne, Record):
            super(Quake, self).append(addOne)
        else:
            raise ValueError('should pass in a Record class')

    def copy(self):
        quake=Quake(loc=self.loc,time=self.time,randID=self.randID,filename=self.filename\
            ,ml=self.ml)
        for record in self:
            quake.append(record.copy())
        return quake

    def set(self,line,mode='byLine'):
        if mode=='byLine':
            return self.setByLine(line)
        elif mode=='fromNDK':
            return self.setFromNDK(line)
        elif mode=='fromIris':
            return self.setFromIris(line)
        elif mode=='byWLX':
            return self.setByWLX(line)
        elif mode=='byMat':
            return self.setByMat(line)
    def setByLine(self,line):
        if len(line) >= 4:
            if len(line)>12:
                self.loc = [float(line[1]), float(line[2]),float(line[-1])]
            else:
                self.loc = [float(line[1]), float(line[2]),0.]
            if len(line)>=14:
                self.ml=float(line[-2])
            self.time = float(line[3])
            self.randID=int(line[9])
            self.filename=line[11]
        return self
    def setFromNDK(self,line):
        sec=float(line[22:26])
        self.time=UTCDateTime(line[5:22]+"0.00").timestamp+sec
        self.loc=[float(line[27:33]),float(line[34:41]),float(line[42:47])]
        m1=float(line[48:55].split()[0])
        m2=float(line[48:55].split()[-1])
        self.ml=max(m1,m2)
        self.filename = self.getFilename()
        return self

    def setFromIris(self,line):
        line=line.split('|')
        self.time=UTCDateTime(line[1]).timestamp
        self.loc[0]=float(line[2])
        self.loc[1]=float(line[3])
        self.loc[2]=float(line[4])
        self.ml=float(line[10])
        self.filename = self.getFilename()
        return self

    def setByWLX(self,line,staInfos=None):
        self.time=UTCDateTime(line[:22]).timestamp
        self.loc=[float(line[23:29]),float(line[30:37]),float(line[38:41])]
        self.ml=float(line[44:])
        self.filename=self.getFilename()
        return self

    def setByMat(self,q):
        pTimeL=q[0].reshape(-1)
        sTimeL=q[1].reshape(-1)
        PS=q[2].reshape(-1)
        self.randID=1
        self.time=matTime2UTC(PS[0])
        self.loc=PS[1:4]
        self.ml=PS[4]
        self.filename=self.getFilename()
        for i in range(len(pTimeL)):
            pTime=0
            sTime=0
            if pTimeL[i]!=0:
                pTime=matTime2UTC(pTimeL[i])
                if sTimeL[i]!=0:
                    sTime=matTime2UTC(sTimeL[i])
                self.append(Record(i,pTime,sTime))
        return self

    def getReloc(self,line):
        self.time=self.tomoTime(line)
        self.loc[0]=float(line[1])
        self.loc[1]=float(line[2])
        self.loc[2]=float(line[3])
        return self

    def tomoTime(self,line):
        m=int(line[14])
        sec=float(line[15])
        return UTCDateTime(int(line[10]),int(line[11]),int(line[12])\
            ,int(line[13]),m+int(sec/60),sec%60).timestamp

    def calCover(self,staInfos):
        '''
        calculate the radiation coverage
        '''
        coverL=np.zeros(360)
        for record in self:
            staIndex= int(record[0])
            Az=DistAz(staInfos[staIndex]['la'],staInfos[staIndex]['lo'],\
                self.loc[0],self.loc[1]).getAz()
            dk=DistAz(staInfos[staIndex]['la'],staInfos[staIndex]['lo'],\
                self.loc[0],self.loc[1]).getDelta()*111.19
            R=int(45/(1+dk/50)+45)
            N=((int(Az)+np.arange(-R,R))%360).astype(np.int64)
            coverL[N]=coverL[N]+1
        L=((np.arange(360)+180)%360).astype(np.int64)
        coverL=np.sign(coverL)*np.sign(coverL[L])*(coverL+coverL[L])
        coverRate=np.sign(coverL).sum()/360
        return coverRate

    def getPTimeL(self,staInfos):
        timePL=np.zeros(len(staInfos))
        for record in self:
            if record.pTime()!=0:
                timePL[record.getStaIndex()]=record.pTime()
        return timePL
        
    def getSTimeL(self,staInfos):
        timeSL=np.zeros(len(staInfos))
        for record in self:
            if record.sTime()!=0:
                timeSL[record.getStaIndex()]=record.sTime()
        return timeSL

    def findTmpIndex(self,staIndex):
        count=0
        for record in self:
            if int(record.getStaIndex())==int(staIndex):
                return count
            count+=1
        return -999

    def setRandIDByMl(self):
        self.randID=int(np.floor(np.abs(10+self.ml)*100));
        namePre=self.filename.split('_')[0]
        self.filename=namePre+'_'+str(self.randID)+'.mat'

    def outputWLX(self):
        Y=getYmdHMSj(UTCDateTime(self.time))
        tmpStr=Y['Y']+'/'+Y['m']+'/'+Y['d']+' '+\
        Y['H']+':'+Y['M']+':'+'%05.2f'%(self.time%60)+\
        ' '+'%6.3f %7.3f %3.1f M %3.1f'%(self.loc[0],\
            self.loc[1],self.loc[2],self.ml)
        return tmpStr

class RecordCC(Record):
    def __init__(self,staIndex=-1, pTime=-1, sTime=-1, pCC=-1, sCC=-1, pM=-1, pS=-1, sM=-1, sS=-1):
        self.append(staIndex)
        self.append(pTime)
        self.append(sTime)
        self.append(pCC)
        self.append(sCC)
        self.append(pM)
        self.append(pS)
        self.append(sM)
        self.append(sS)

    def getPCC(self):
        return self[3]
    def getSCC(self):
        return self[4]
    def getPM(self):
        return self[5]
    def getPS(self):
        return self[6]
    def getSM(self):
        return self[7]
    def getSS(self):
        return self[8]
    def getPMul(self):
        return (self.getPCC()-self.getPM())/self.getPS()
    def getSMul(self):
        return (self.getSCC()-self.getSM())/self.getSS()  


class QuakeCC(Quake):
    '''
    expand the basic class Quake for storing the quake result 
    of MFT and WMFT
    the basic information include more: cc,M,S,tmpName
    '''
    def __init__(self, cc=-9, M=-9, S=-9, loc=[-999, -999,10],  time=-1, randID=None, \
        filename=None,tmpName=None,ml=None):
        super(Quake, self).__init__()
        self.cc=cc
        self.M=M
        self.S=S
        self.loc = [0,0,0]
        self.loc[:len(loc)]=loc
        self.time = time
        self.tmpName=tmpName
        self.ml=ml
        if randID != None:
            self.randID=randID
        else:
            self .randID=int(10000*np.random.rand(1))+10000*2
        if filename != None:
            self.filename = filename
        else:
            self.filename = self.getFilename()

    def append(self, addOne):
        if isinstance(addOne, RecordCC):
            super(Quake, self).append(addOne)
        else:
            raise ValueError('should pass in a RecordCC class')
    def getMul(self):
        return (self.cc-self.M)/self.S

    def calCover(self,staInfos,minCC=0.5):
        '''
        calculate the radiation coverage which have higher CC than minCC
        '''
        coverL=np.zeros(360)
        for record in self:
            if (record.pTime()>0 or record.sTime())>0 \
            and(record.getPCC()>minCC or record.getSCC()>minCC): 
                staIndex= int(record[0])
                Az=DistAz(staInfos[staIndex]['la'],staInfos[staIndex]['lo'],\
                    self.loc[0],self.loc[1]).getAz()
                dk=DistAz(staInfos[staIndex]['la'],staInfos[staIndex]['lo'],\
                    self.loc[0],self.loc[1]).getDelta()*111.19
                R=int(45/(1+dk/50)+45)
                N=((int(Az)+np.arange(-R,R))%360).astype(np.int64)
                coverL[N]=coverL[N]+1
        L=((np.arange(360)+180)%360).astype(np.int64)
        coverL=np.sign(coverL)*np.sign(coverL[L])*(coverL+coverL[L])
        coverRate=np.sign(coverL).sum()/360
        return coverRate
    def summary(self,count=0):
        ml=-9
        if self.ml!=None:
            ml=self.ml
        Summary='quake: %f %f %f num: %d index: %d randID: %d filename: \
            %s %s %f %f %f %f %f\n' % (self.loc[0], self.loc[1],\
            self.time, len(self),count, self.randID, \
            self.filename,str(self.tmpName),self.cc,self.M,self.S,ml,self.loc[2])
        return Summary
    def setByLine(self,line):
        if len(line) >= 4:
            self.loc = [float(line[1]), float(line[2]),float(line[-1])]
            if len(line)>=18:
                self.ml=float(line[-2])
                self.S=float(line[-3])
                self.M=float(line[-4])
                self.cc=float(line[-5])
                self.tmpName=line[-6]
            self.time = float(line[3])
            self.randID=int(line[9])
            self.filename=line[11]
        return self
    def setByMat(self,q):
        '''
        0('tmpIndex', 'O'), ('name', 'O'), ('CC', 'O'), 
        3('mean', 'O'), ('std', 'O'), ('mul', 'O'), 
        6('pCC', 'O'), ('sCC', 'O'), ('pM', 'O'), 
        9('sM', 'O'), ('pS', 'O'), ('sS', 'O'), 
        12('PS', 'O'), ('pTime', 'O'), ('sTime', 'O'),
        15('pD', 'O'), ('sD', 'O'), ('tmpTime', 'O'), 
        18('oTime', 'O'), ('eTime', 'O')])
        '''
        self.cc=q[2][0,0]
        self.S=q[4][0,0]
        self.M=q[3][0,0]
        self.tmpName=str(q[1][0])
        pTimeL=q[13].reshape(-1)
        sTimeL=q[14].reshape(-1)
        pCC=q[6].reshape(-1)
        sCC=q[7].reshape(-1)
        pM=q[8].reshape(-1)
        sM=q[9].reshape(-1)
        pS=q[10].reshape(-1)
        sS=q[11].reshape(-1)
        PS=q[12].reshape(-1)
        self.time=matTime2UTC(PS[0])
        self.loc=PS[1:4]
        self.ml=PS[4]
        self.filename=self.getFilename()
        for i in range(len(pTimeL)):
            pTime=0
            sTime=0
            if pTimeL[i]!=0:
                pTime=matTime2UTC(pTimeL[i])
                if sTimeL[i]!=0:
                    sTime=matTime2UTC(sTimeL[i])
                self.append(RecordCC(i,pTime,sTime,pCC[i],sCC[i],pM[i],pS[i],sM[i],sS[i]))
        return self
    def setByWLX(self,line,tmpNameL=None):
        lines=line.split()
        self.time=UTCDateTime(lines[1]+' '+lines[2]).timestamp
        self.loc=[float(lines[3]),float(lines[4]),float(lines[5])]
        self.ml=float(lines[6])
        self.cc=float(lines[7])
        tmpStr=lines[10]
        tmpTime=UTCDateTime(int(tmpStr[0:4]),int(tmpStr[4:6]),int(tmpStr[6:8]),int(tmpStr[8:10]),\
                int(tmpStr[10:12]),int(tmpStr[12:14])).timestamp
        tmpName=str(int(tmpTime))
        if tmpNameL!=None:       
            for tmpN in tmpNameL:
                if tmpName == tmpN.split('/')[-1].split('_')[0]:
                    tmpName=tmpN
        self.tmpName=tmpName
        self.filename=self.getFilename()
        return self

def removeBadSta(quakeLs,badStaLst=[]):
    for quakeL in quakeLs:
        for quake in quakeL:
            for record in quake:
                if record.getStaIndex() in badStaLst:
                    record[1]=0
                    record[2]=0
                    print("setOne")

def getQuakeLD(quakeL):
    D={}
    for quake in quakeL:
        D[quake.filename]=quake 
    return D

def divideRange(L, N):
    dL = (L[1]-L[0])/N
    subL = np.arange(0, N+1)*dL+L[0]
    dR = {'minL': subL[0:-1], 'maxL': subL[1:], 'midL': \
    (subL[0:-1]+subL[1:])/2}
    return dR

def findLatterOne(t0, tL):
    indexL=np.where(tL>t0)
    if indexL.shape[0]>0:
        return indexL[0], tL(indexL[0])
    return -1, -1

class arrival(object):
    def __init__(self, time):
        self.time = time


class quickTaupModel:
    '''
    pre-calculated taup model for quick usage
    '''
    def __init__(self, modelFile='iaspTaupMat'):
        matload = sio.loadmat(modelFile)
        self.interpP = interp.interp2d(matload['dep'].reshape([-1]),\
            matload['deg'].reshape([-1]), matload['taupMatP'])
        self.interpS = interp.interp2d(matload['dep'].reshape([-1]),\
            matload['deg'].reshape([-1]), matload['taupMatS'])
        pTime0 = matload['taupMatP'][:,0].reshape([-1])
        sTime0 = matload['taupMatS'][:,0].reshape([-1])
        dTime = sTime0-pTime0
        dL = np.argsort(dTime)
        self.interpO = interp.interp1d(dTime[dL], pTime0[dL], \
            fill_value='extrapolate')

    def get_travel_times(self,dep, deg, phase_list='p'):
        if phase_list[0]=='p':
            a = arrival(self.interpP(dep, deg)[0])
        else:
            a = arrival(self.interpS(dep, deg)[0])
        return [a]

    def get_orign_times(self, pIndex, sIndex, delta):
        return pIndex-self.interpO((sIndex-pIndex)*delta)/delta

def getQuakeInfoL(quakeL,loc0=np.array([37.8,140,0])):
    PS=np.zeros((len(quakeL),5))
    for i in range(len(quakeL)):
        PS[i,0]=quakeL[i].time
        PS[i,1:4]=quakeL[i].loc-loc0
        PS[i,4]=quakeL[i].ml
    return PS

def saveQuakeLs(quakeLs, filename,mod='o'):
    with open(filename, 'w') as f:
        if mod=='o':
            count = 0
            for quakeL in quakeLs:
                f.write('day\n')
                for quake in quakeL:
                    f.write(quake.summary(count))
                    count +=1
                    for record in quake:
                        f.write(record.summary())
        if mod=='ML':
            for quake in quakeLs:
                f.write(quake.outputWLX()+'\n')

def getStaNameIndex(staInfos):
    staNameIndex=dict()
    for i in range(len(staInfos)):
        staNameIndex.update({staInfos[i]['sta']: i})
    return staNameIndex

def readQuakeLs(filenames, staInfos, mode='byQuake', \
    N=200, dH=0, isQuakeCC=False,key=None,minMul=8,\
    tmpNameL=None):
    '''
    read quakeLst in different form to differen form
    '''
    def getQuakeLFromDay(day):
        quakeL=list()
        for q in day:
            if not isQuakeCC:
                quake=Quake().set(q,'byMat')
            else:
                quake=QuakeCC().set(q,'byMat')
            if quake.time>0:
                quakeL.append(quake)
        return quakeL

    if mode=='byMatDay':
        dayMat=sio.loadmat(filenames)
        if key ==None:
            key=dayMat.keys()[-1]
        dayMat=dayMat[key][-1]
        quakeLs=list()
        for day in dayMat:
            day=day[-1][-1]
            quakeLs.append(getQuakeLFromDay(day))
        return quakeLs

    if mode=='byMat':
        dayMat=sio.loadmat(filenames)
        if key == None:
            key=dayMat.keys()[-1]
        day=dayMat[key][-1]
        return getQuakeLFromDay(day)
        
    if mode=='byWLX':
        quakeL=[]
        with open(filenames) as f:
            if not isQuakeCC:
                for line in f.readlines():
                    quakeL.append(Quake().set(line,'byWLX'))
            else:
                for line in f.readlines()[1:]:
                    quakeL.append(QuakeCC().set(line,'byWLX'))
        return quakeL

    with open(filenames) as f:
        lines = f.readlines()
        quakeLs = list()
        if mode == 'byQuake':
            for line in lines:
                line = line.split()
                if line[0] == 'day':
                    quakeL = list()
                    quakeLs.append(quakeL)
                    continue
                if line[0].split()[0] == 'quake:':
                    if isQuakeCC:
                        quake=QuakeCC()
                    else:
                        quake = Quake()
                    quake.set(line,'byLine')
                    quakeL.append(quake)
                    continue
                if isQuakeCC:
                    record=RecordCC()
                else:
                    record=Record()
                quake.append(record.set(line,'byLine'))
            return quakeLs
        if mode == 'bySta':
            staArrivalP = [[] for i in range(N)]
            staArrivalS = [[] for i in range(N)]
            maxStaCount = 0
            for line in lines:
                line = line.split()
                if line[0] == 'day':
                    continue
                if line[0].split()[0] == 'quake:':
                    continue
                staIndex = int(line[0])
                maxStaCount = max(staIndex+1, maxStaCount)
                timeP = float(line[1])
                timeS = float(line[2])
                if timeP > 1:
                    staArrivalP[staIndex].append(timeP)
                if timeS > 1:
                    staArrivalS[staIndex].append(timeS)
            return staArrivalP[0:maxStaCount], staArrivalS[0:maxStaCount]

        if mode == 'SC':
            staNameIndex = getStaNameIndex(staInfos)
            staArrivalP = [[] for i in range(len(staInfos))]
            staArrivalS = [[] for i in range(len(staInfos))]
            for line in lines:
                if line[0] == '2':
                    continue
                lineCell = line.split(',')
                time=UTCDateTime(lineCell[1]).timestamp+dH*3600
                staIndex = staNameIndex[lineCell[0].strip()]
                if lineCell[2][0] == 'P':
                    staArrivalP[staIndex].append(time)
                else:
                    staArrivalS[staIndex].append(time)
            return staArrivalP, staArrivalS
        if mode == 'NDK':
            quakeLs=[]
            time0=0
            for i in range(0,len(lines),5):
                line=lines[i]
                quake=Quake()
                quake=quake.set(line,'fromNDK')
                time1=int(quake.time/86400)
                print(time1)
                if time1>time0:
                    quakeLs.append([])
                    time0=time1
                quakeLs[-1].append(quake)
            return quakeLs
        if mode=='IRIS':
            quakeLs=[]
            time0=0
            for i in range(0,len(lines)):
                line=lines[i]
                quake=Quake()
                quake=quake.set(line,'fromIris')
                time1=int(quake.time/86400)
                print(time1)
                if time1>time0:
                    quakeLs.append([])
                    time0=time1
                quakeLs[-1].append(quake)
            return quakeLs



def readQuakeLsByP(filenamesP, staInfos, mode='byQuake',  N=200, dH=0,key=None\
    ,isQuakeCC=False):
    quakeLs=[]
    for file in glob(filenamesP):
        quakeLs=quakeLs+readQuakeLs(file, staInfos, mode=mode,  N=N, dH=dH,\
            key=key,isQuakeCC=isQuakeCC)
    return quakeLs

def compareTime(timeL, timeL0, minD=2):
    timeL = np.sort(np.array(timeL))
    timeL0 = np.sort(np.array(timeL0))
    dTime = list()
    count0 = 0
    count = 0
    N = timeL.shape[0]
    N0 = timeL0.shape[0]
    i = 0
    for time0 in timeL0:
        if i == N-1:
            break
        if time0 < timeL[0]:
            continue
        if time0 > timeL[-1]:
            break
        count0 += 1
        for i in range(i, N):
            if abs(timeL[i]-time0)<minD:
                dTime.append(timeL[i]-time0)
                count += 1
            if i == N-1:
                break
            if timeL[i] > time0:
                break
    return dTime, count0, count

def getSA(data):
    data=data-data.mean()
    data0=data*0
    for i in range(1,data.shape[0]):
        data0[i,:]=data0[i-1,:]+data[i-1,:]
    return data0.max()

def saveQuakeWaveform(staL, quake, quakeIndex, matDir='output/'\
    ,index0=-500,index1=500,dtype=np.float32):
    indexL = np.arange(index0, index1)
    iNum=indexL.size
    fileName = matDir+'/'+quake.filename
    loc=quake.loc
    dayDir=os.path.dirname(fileName)
    if not os.path.exists(dayDir):
        os.mkdir(dayDir)
    pWaveform = np.zeros((len(quake), iNum, 3),dtype=dtype)
    sWaveform = np.zeros((len(quake), iNum, 3),dtype=dtype)
    staIndexL = np.zeros(len(quake))
    pTimeL = np.zeros(len(quake))
    sTimeL = np.zeros(len(quake))
    deltaL = np.zeros(len(quake))
    ml=0
    sACount=0
    for i in range(len(quake)):
        record = quake[i]
        staIndex = record.getStaIndex()
        pTime = record.pTime()
        sTime = record.sTime()
        staIndexL[i] = staIndex
        pTimeL[i] = pTime
        sTimeL[i] = sTime
        if pTime != 0:
            try:
                pWaveform[i, :, :] = staL[staIndex].data.getDataByTimeLQuick\
                (pTime + indexL*staL[staIndex].data.delta)
            except:
                print("wrong p wave")
            else:
                pass
            deltaL[i] = staL[staIndex].data.delta
        if sTime != 0:
            try:
                sWaveform[i, :, :] = staL[staIndex].data.getDataByTimeLQuick\
                (sTime + indexL*staL[staIndex].data.delta)
            except:
                print("wrong s wave")
            else:
                deltaL[i]= staL[staIndex].data.delta
                dk=DistAz(staL[staIndex].loc[0],staL[staIndex].loc[1],loc[0],loc[1]).getDelta()*111.19
                sA=getSA(sWaveform[i, :, :])*staL[staIndex].data.delta
                ml=ml+np.log10(sA)+1.1*np.log10(dk)+0.00189*dk-2.09-0.23
                sACount+=1
    if sACount==0:
        ml=-999
    else:
        ml/=sACount
    sio.savemat(fileName, {'time': quake.time, 'loc': quake.loc, \
        'staIndexL': staIndexL, 'pTimeL': pTimeL, 'pWaveform': \
        pWaveform, 'sTimeL': sTimeL, 'sWaveform': sWaveform, \
        'deltaL': deltaL, 'indexL': indexL,'ml':ml})
    return ml

def resampleWaveform(waveform,n):
    b,a=signal.bessel(2,1/n*0.8)
    waveform['deltaL']=waveform['deltaL']*n
    waveform['indexL']=[waveform['indexL'][0][0:-1:n]]
    N=waveform['indexL'][0].size
    waveform['pWaveform']=signal.resample(signal.filtfilt(\
        b,a,waveform['pWaveform'][:,0:N*n,:],axis=1),N,axis=1)
    waveform['sWaveform']=signal.resample(signal.filtfilt(\
        b,a,waveform['sWaveform'][:,0:N*n,:],axis=1),N,axis=1)
    return waveform

def resampleWaveformL(waveformL,n):
    for waveform in waveformL:
        waveform=resampleWaveform(waveform,n)
    return waveformL


def getMLFromWaveform(quake, staInfos, matDir='output/',minSACount=3):
    filename=matDir+'/'+quake.filename
    waveform=sio.loadmat(filename)
    ml=0
    loc=waveform['loc'][0]
    sACount=0
    if len(waveform['staIndexL'])<=0:
        ml=-999
        print('wrong ml')
        return ml
    for i in range(len(waveform['staIndexL'][0])):
        if waveform['sTimeL'][0][i]!=0:
            staIndex=int(waveform['staIndexL'][0][i])
            dk=DistAz(staInfos[staIndex]['la'],\
                staInfos[staIndex]['lo'],loc[0],loc[1]).getDelta()*111.19
            if dk<30 or dk>300:
                continue
            sA=getSA(waveform['sWaveform'][i, :, :])*waveform['deltaL'][0][i]
            ml=ml+np.log10(sA)+1.1*np.log10(dk)+0.00189*dk-2.09-0.23
            sACount+=1
    if sACount<minSACount:
        ml=-999
    else:
        ml/=sACount
    quake.ml=ml
    print(sACount,ml)
    return ml

def getMLFromWaveformL(quakeL, staInfos, matDir='output/',isQuick=False):
    count=0
    for quake in quakeL:
        count+=1
        print(count)
        if quake.ml!=None and isQuick:
            if quake.ml >-2 and quake.ml<3:
                continue
        quake.ml=getMLFromWaveform(quake,staInfos,matDir=matDir)

def getMLFromWaveformLs(quakeLs, staInfos, matDir='output/'):
    for quakeL in quakeLs:
        for quake in quakeL:
            quake.ml=getMLFromWaveform(quake,staInfos,matDir=matDir)
            
def saveQuakeLWaveform(staL, quakeL, matDir='output/',\
    index0=-500,index1=500,dtype=np.float32):
    if not os.path.exists(matDir):
        os.mkdir(matDir)
    for i in range(len(quakeL)):
         quakeL[i].ml=saveQuakeWaveform(staL, quakeL[i], i,\
          matDir=matDir,index0=index0,index1=index1,dtype=dtype)

def loadWaveformByQuake(quake,matDir='output',isCut=False,index0=-250,index1=250,f=[-1,-1]):
    fileName = matDir+'/'+quake.filename
    waveform=sio.loadmat(fileName)
    if isCut:
        i0=np.where(waveform['indexL'][0]==index0)[0][0]
        i1=np.where(waveform['indexL'][0]==index1)[0][0]
        waveform['pWaveform']=waveform['pWaveform'][:,i0:i1,:].astype(np.float32)
        waveform['sWaveform']=waveform['sWaveform'][:,i0:i1,:].astype(np.float32)
        waveform['indexL']=[waveform['indexL'][0][i0:i1]]
    if f[0]>0:
        f0=0.5/waveform['deltaL'].max()
        b, a = signal.butter(8, [f[0]/f0,f[1]/f0], 'bandpass')
        waveform['pWaveform']=signal.filtfilt(b,a,waveform['pWaveform'],axis=1)
        waveform['sWaveform']=signal.filtfilt(b,a,waveform['sWaveform'],axis=1)
    return waveform

def loadWaveformLByQuakeL(quakeL,matDir='output',isCut=False,index0=-250,index1=250,f=[-1,-1]):
    waveformL=[]
    tmpNameL=[]
    for quake in quakeL:
        tmpNameL.append(quake.filename)
        waveformL.append(loadWaveformByQuake(quake,matDir=matDir,isCut=isCut,index0=index0,index1=index1,f=f))
    return waveformL, tmpNameL
    

def genTaupTimeM(model='iasp91', N=6, matFile='iaspTaupMat',depN=200, degN=4000):
    managers = Manager()
    resL = [managers.list() for i in range(N)]
    taupDict = {'dep': None,'deg': None, 'taupMatP': None, 'taupMatS':None}
    taupDict['deg'] = np.power(np.arange(degN)/degN,2)*180
    taupDict['dep'] = np.concatenate([np.arange(depN/2),np.arange(depN/2)*10+depN/2])
    taupDict['taupMatP']= np.zeros((degN, depN))
    taupDict['taupMatS']= np.zeros((degN, depN))
    taupM = taup.TauPyModel(model=model)
    processL=list()
    for i in range(N):
        processTmp = Process(target=_genTaupTimeM, args=(taupDict, i, N, taupM, resL[i]))
        processTmp.start()
        processL.append(processTmp)

    for processTmp in processL:
        processTmp.join()

    for index in range(N):
        i=0
        for depIndex in range(index, depN, N):
            for degIndex in range(degN):
                taupDict['taupMatP'][degIndex, depIndex]=resL[index][i][0]
                taupDict['taupMatS'][degIndex, depIndex]=resL[index][i][1]
                i += 1
    sio.savemat(matFile, taupDict)
    return taupDict



def _genTaupTimeM(taupDict, index, N, taupM, resL):
    depN = len(taupDict['dep'][:])
    degN = len(taupDict['deg'][:])
    for depIndex in range(index, depN, N):
        print(depIndex)
        dep = taupDict['dep'][depIndex]
        for degIndex in range(degN):
            deg = taupDict['deg'][degIndex]
            if degIndex==0:
                print(depIndex, degIndex, getEarliest(taupM.get_travel_times\
                    (dep, deg, ['p', 'P', 'PP', 'pP'])))
            resL.append([getEarliest(taupM.get_travel_times(dep, deg, \
                ['p', 'P', 'PP', 'pP'])), getEarliest(taupM.get_travel_times\
            (dep, deg, ['s', 'S', 'SS', 'sS']))])

def getEarliest(arrivals):
        time=10000000
        if len(arrivals)==0:
            print('no phase')
            return 0
        for arrival in arrivals:
            time = min(time, arrival.time)
        return time

def validMean(vL):
    vM=np.median(vL)
    vD=vL.std()
    vLst=np.where(np.abs(vL-vM)<10*vD)
    vL=vL[vLst]
    return vL

'''
tool for read data recorder logs
to get location from gps info
getLocByLog
getLocByLogs
getLocByLogsP
'''
#GPS: POSITION: N41:45:04.50 E103:23:45.46
def getLocByLog(filename):
    p=r"GPS: POSITION.{36}"
    if not os.path.exists(filename):
        return 999,999,999,999,999,999
    with open(filename) as f:
        lines=f.read()
        pRe=re.compile(p)
        laL=[]
        loL=[]
        zL=[]
        for line in pRe.findall(lines):
            EW=1
            NS=1
            if line[15]=='S':
                NS=-1
            if line[28]=='W':
                EW=-1
            la=NS*(float(line[16:18])+float(line[19:21])/60+float(line[22:27])/3600)
            laL.append(la)
            lo=EW*(float(line[29:32])+float(line[33:35])/60+float(line[36:41])/3600)
            loL.append(lo)
            zL.append(float(line[42:-1]))
    if len(laL)>0 and len(loL)>0:
        laL=np.array(laL)
        loL=np.array(loL)
        zL=np.array(zL)
        laL=validMean(laL)
        loL=validMean(loL)
        zL=validMean(zL)
    if len(laL)>0 and len(loL)>0 and len(zL)>0:
        return laL.mean(), loL.mean(), laL.std(), loL.std(), zL.mean(), zL.std()
    else:
        return 999, 999, 999, 999, 999, 999

def getLocByLogs(filenames):
    laL=[]
    loL=[]
    zL=[]
    for filename in filenames:
        la, lo, laD, loD, z, zD = getLocByLog(filename)
        if laD>1e-3 or loD>1e-3:
            print('RMS too large', laD, loD)
            continue
        if la !=999 and lo!=999:
            laL.append(la)
            loL.append(lo)
            zL.append(z)
    if len(laL)>0 and len(loL)>0:
        laL=np.array(laL)
        loL=np.array(loL)
        zL=np.array(zL)
        return laL.mean(),loL.mean(),laL.std(),loL.std(),zL.mean(),zL.std()
    else:
        return 999,999,999,999,999,999

def getLocByLogsP(p):
    filenames=[];
    for file in glob(p):
        filenames.append(file)
    return getLocByLogs(filenames)

'''
this part is designed for getting sta info (loc and file path)
'''
def getStaAndFileLst(dirL,filelstName,staFileName):
    def writeMFileInfo(f,mFile,dayDir,staName):
        comp=['BH','BH','BH']
        fileIndex=mFile[0:6]+'_'+comp[int(mFile[-3])-1]
        f.write("%s %s %s\n"%(staName,fileIndex,dayDir))
    staLst={}
    with open(filelstName,'a') as f:
        for Dir in dirL:
            for tmpDir in glob(Dir+'/[A-Z]*'):
                try:
                    if not os.path.isdir(tmpDir):
                        continue
                    for staName in os.listdir(tmpDir):
                        staDir=tmpDir+'/'+staName+'/'
                        if not os.path.isdir(staDir):
                            continue
                        staLogsP=staDir+'*log'
                        try:
                            la,lo,laD,loD,z,zD = getLocByLogsP(staLogsP)
                        except:
                            print('wrong')
                            continue
                        else:
                            pass
                        print(tmpDir,staName,la,lo,laD,loD,z,zD)
                        if la !=999 and lo!=999:
                            if staName in staLst:
                                if laD+loD < staLst[staName][2]+staLst[staName][3]:
                                    staLst[staName]=[la,lo,laD,loD,tmpDir,z,zD]
                            else:
                                staLst[staName]=[la,lo,laD,loD,tmpDir,z,zD]
                        continue
                        for dayDir in glob(staDir+'R*'):
                            for hourDir in glob(dayDir+'/'+'00'):
                                for mFile in glob(hourDir+'/*1.m'):
                                    mFile=mFile.split('/')[-1]
                                    writeMFileInfo(f,mFile,dayDir,staName)
                except:
                    print("***********errro*********")
                else:
                    pass
    with open(staFileName,'w+') as staFile :
        for staName in staLst:
            staFile.write("hima %s BH %f %f %f %f %f %f\n"%(staName,staLst[staName][1], \
                staLst[staName][0],staLst[staName][3],staLst[staName][2],\
                staLst[staName][5],staLst[staName][6]))

def getThem():
    fileL=['/media/jiangyr/XIMA_I/XIMA_I/','/media/jiangyr/XIMA_II/','/media/jiangyr/XIMA_III/XIMA_III/',\
            '/media/jiangyr/XIMA_IV/XIMA_IV/','/media/jiangyr/XIMA_V/XIMA_V/']
    getStaAndFileLst(fileL,'fileLst','staLst')

def checkFile(filename):
    sta={}
    with open(filename) as f:
        for line in f.readlines():
            staIndex=line.split(' ')[1].split('/')[-1]
            if staIndex in sta:
                print(line)
                print(sta[staIndex])
            else:
                sta[staIndex]=line

def loadFileLst(staInfos,filename):
    staDict={}
    for staTmp in staInfos:
        staDict.update({staTmp['sta']:{}})
    with open(filename) as f:
        for line in f.readlines():
            infos=line.split()
            if infos[0] in staDict:
                staDict[infos[0]].update({infos[1]:infos[2]})
    return staDict

def getStaInArea(staInfos,fileNew,R):
    with open(fileNew,'w+') as f:
        for staInfo in staInfos:
            if staInfo['la']>=R[0] and \
                    staInfo['la']<=R[1] and \
                    staInfo['lo']>=R[2] and \
                    staInfo['lo']<=R[3]:
                f.write("%s %s %s %f %f 0 0 %f\n"%(staInfo['net'],staInfo['sta'],\
                    staInfo['comp'][0][0:2],staInfo['lo'],staInfo['la'],staInfo['dep']))


def getStaInfoFromSac(sacDir, staInfoFile='staLstSac',staInfos=[],\
    dataDir='dataDis/',R=[33,44,106,116]):
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
    staLst={}
    for staInfo in staInfos:
        staLst[staInfo['sta']]=[staInfo['la'],staInfo['lo']]
    with open(staInfoFile,'a') as f:
        for sacDirTmp in glob(sacDir+'/20*00/'):
            for sacFile in glob(sacDirTmp+'/*.BHE'):
                fileName=sacFile.split('/')[-1]
                net,station=fileName.split('.')[0:2]
                print(net,station)
                if station not in staLst:
                    sac=obspy.read(sacFile)[0].stats.sac
                    print("%s %s %s %f %f 0 0 %f\n"%('hima',station,\
                        'BH',sac['stlo'],sac['stla'],sac['stel']))
                    f.write("%s %s %s %f %f 0 0 %f\n"%('hima',station,\
                        'BH',sac['stlo'],sac['stla'],sac['stel']))
                    staLst[station]=[sac['stla'],sac['stlo']]
                plt.plot(staLst[station][1],staLst[station][0],'.r')
            figName=dataDir+sacDirTmp.split('/')[-2]+'.png'
            plt.xlim(R[2:])
            plt.ylim(R[:2])
            plt.savefig(figName)
            plt.close()

def toTmpNameD(tmpNameL):
    tmpNameD={}
    for i in range(len(tmpNameL)):
        tmpNameD[tmpNameL[i]]=i
    return tmpNameD

def synQuake(staInfos,loc,indexL=[],N=20,modelFile='taupTimeMat.mat',\
    oTime=0,isS=True,ml=-9):
    quake=Quake(time=oTime)
    quake.loc=loc
    quake.ml=ml
    if isinstance(modelFile,str):
        timeM=quickTaupModel(modelFile)
    else:
        timeM=modelFile
    if len(indexL)==0:
        indexL=np.floor(np.random.rand(N)*len(staInfos)).astype(np.int64)
    for index in indexL:
        staLa=staInfos[index]['la']
        staLo=staInfos[index]['lo']
        dep=staInfos[index]['dep']/1000+loc[2]
        delta=DistAz(quake.loc[0],quake.loc[1],\
                    staLa,staLo).delta
        timeP=timeM.get_travel_times(dep,delta,'p')[0].time+oTime
        timeS=0
        if isS:
            timeS=timeM.get_travel_times(dep,delta,'s')[0].time+oTime
        quake.append(Record(index,timeP,timeS))
    return quake

def synQuakeV2(quake,staInfos,indexL=[],N=20,modelFile='taupTimeMat.mat',\
   isS=True):
    quake.clear()
    loc=quake.loc
    oTime=quake.time
    if isinstance(modelFile,str):
        timeM=quickTaupModel(modelFile)
    else:
        timeM=modelFile
    if len(indexL)==0:
        indexL=np.floor(np.random.rand(N)*len(staInfos)).astype(np.int64)
    for index in indexL:
        staLa=staInfos[index]['la']
        staLo=staInfos[index]['lo']
        dep=staInfos[index]['dep']/1000+loc[2]
        delta=DistAz(quake.loc[0],quake.loc[1],\
                    staLa,staLo).delta
        timeP=timeM.get_travel_times(dep,delta,'p')[0].time+oTime
        timeS=0
        if isS:
            timeS=timeM.get_travel_times(dep,delta,'s')[0].time+oTime
        if not isinstance(quake,QuakeCC):
            quake.append(Record(index,timeP,timeS))
        else:
            quake.append(RecordCC(index,timeP,timeS))
    return quake

def analysis(quakeLs,staInfos,outDir='fig/',minSta=6,maxDep=80,\
    bTime=UTCDateTime(2014,1,1).timestamp,eTime=UTCDateTime(2017,10,1).timestamp):
    dayNum=int((eTime-bTime)/86400)
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    mlL=[]
    timeL=[]
    #staNum=np.zeros(len(staInfos))
    staDayNum=np.zeros((len(staInfos),dayNum))
    for quakeL in quakeLs:
        for quake in quakeL:
            if len(quake)<minSta:
                continue
            if quake.loc[2]>maxDep:
                continue
            if quake.ml==None:
                continue
            if quake.ml<-5:
                continue
            mlL.append(quake.ml)
            timeL.append(quake.time)
            dayIndex=int((quake.time-bTime)/86400)
            for record in quake:
                staDayNum[record.getStaIndex(),dayIndex]+=1
    #plt.subplot(2,2,1)
    plt.hist(np.array(mlL),bins=40)
    plt.title('ml dis')
    plt.savefig(outDir+'mlDis.png')
    plt.xlabel('ml')
    plt.ylabel('count')
    plt.close()

    #plt.subplot(2,2,2)
    #plt.hist(staDayNum.sum(axis=1),bins=20)
    plt.bar(np.arange(len(staInfos)),staDayNum.sum(axis=1))
    plt.xlabel('sta Index')
    plt.ylabel('record count')
    plt.title('station record num')
    plt.savefig(outDir+'staRecordNum.png')
    plt.close()
    
    #plt.subplot(2,2,3)
    i=0
    for staInfo in staInfos:
        plt.plot(staInfo['lo'],staInfo['la'],'^b', markersize=np.log(1+np.sign(staDayNum[i,:]).sum()))
        i+=1
    plt.title('station dis with date num')
    plt.savefig(outDir+'staDis.png')
    plt.close()

    #plt.subplot(2,2,4)
    i=0
    plt.pcolor(np.sign(staDayNum).transpose())
    plt.xlabel('sta Index')
    plt.ylabel('date from '+UTCDateTime(bTime).strftime('%Y%m%d'))
    plt.title('sta date')
    plt.savefig(outDir+'staDate.png')
    plt.close()

    pTimeL=[]
    sTimeL=[]
    depL=[]
    for quakeL in quakeLs:
        for quake in quakeL:
            for record in quake:
                pTime=record.pTime()
                sTime=record.sTime()
                if sTime > 0:
                    pTimeL.append(pTime-quake.time)
                    sTimeL.append(sTime-quake.time)
                    depL.append(quake.loc[2])
    pTimeL=np.array(pTimeL)
    sTimeL=np.array(sTimeL)
    depL=np.array(depL)
    #plt.plot(pTimeL,sTimeL,'.',markersize=0.02,markerfacecolor='blue',markeredgecolor='blue',alpha=0.8)
    plt.scatter(pTimeL,sTimeL,0.01,depL,alpha=0.5,marker=',')
    plt.title('pTime-sTime')
    plt.xlabel('pTime')
    plt.ylabel('sTime')
    plt.savefig(outDir+'pTime-sTime.png',dpi=300)
    plt.close()


def calSpec(x,delta):
    spec=np.fft.fft(x)
    N=x.size
    fL=np.arange(N)/N*1/delta
    return spec,fL

def plotSpec(waveform,isNorm=True,plotS=False,alpha=0.1):
    cL="rgb"
    N=1
    if plotS:
        N=2
    for i in range(len(waveform['staIndexL'][0])):
        if waveform['pTimeL'][0][i]!=0:
            for comp in range(3):
                W=waveform['pWaveform'][i,:,comp]
                print(N)
                plt.subplot(N,3,comp+1)

                if W.max()<=0:
                    continue
                W=W/np.linalg.norm(W)
                spec,fL=calSpec(W,waveform['deltaL'][0,i])
                plt.plot(fL,abs(spec),cL[comp],alpha=alpha)
                plt.xlim([0,fL[-1]/2])
                if plotS:
                    plt.subplot(N,3,3+comp+1)
                else:
                    continue
                W=waveform['sWaveform'][i,:,comp]
                if W.max()<=0:
                    continue
                W=W/np.linalg.norm(W)
                spec,fL=calSpec(W,waveform['deltaL'][0,i])
                plt.plot(fL,abs(spec),cL[comp],alpha=alpha)
                plt.xlim([0,fL[-1]/2])
    plt.show()

def selectQuakeByDis(quakeLs,R,staInfos,minDis=0,maxDis=20,outDir='output/'\
    ,bTime=UTCDateTime(1970,1,1).timestamp,\
    eTime=UTCDateTime(2100,1,1).timestamp,minMl=5):
    midLa=(R[0]+R[1])/2
    midLo=(R[2]+R[3])/2
    quakeLNew=[]
    for quakeL in quakeLs:
        for quake in quakeL:
            if quake.time<bTime or quake.time>eTime:
                continue
            if quake.ml<minMl:
                continue
            delta=DistAz(midLa,midLo,quake.loc[0],quake.loc[1]).getDelta()
            if delta<minDis or delta>maxDis:
                continue
            quakeLNew.append(quake)
    return quakeLNew

def selectQuake(quakeLs,R,staInfos,minSta=10,laN=30,loN=30,maxCount=25,minCover=0.8,\
    maxDep=60,isF=True,outDir='output/'):
    quakeL=[]
    quakeNumL=[]
    laL=np.arange(R[0],R[1],(R[1]-R[0])/laN)
    loL=np.arange(R[2],R[3],(R[3]-R[2])/loN)
    aM=np.zeros((laN+1,loN+1))  
    for quakeLTmp in quakeLs:
        for quake in quakeLTmp:
            num=0
            for record in quake:
                if record.pTime()>0:
                    num+=1
            #num=len(quake)
            if num < minSta:
                continue
            if quake.calCover(staInfos)<minCover:
                continue
            if quake.loc[0]<R[0] or quake.loc[0]>R[1] or quake.loc[1]<R[2] or quake.loc[1]>R[3]:
                continue
            if quake.loc[2]>maxDep:
                continue
            if not os.path.exists(outDir+quake.filename):
                continue
            quakeL.append(quake)
            quakeNumL.append(num)
    L=np.argsort(-np.array(quakeNumL))
    quakeLNew=[]
    for i in L:
        quake=quakeL[i]
        laIndex=np.argmin(np.abs(quake.loc[0]-laL))
        loIndex=np.argmin(np.abs(quake.loc[1]-loL))
        if aM[laIndex][loIndex]>=maxCount:
            continue
        aM[laIndex][loIndex]+=1
        quakeLNew.append(quake)
    return quakeLNew

def selectRecord(quake,maxDT=35):
    for record in quake:
        if record.pTime()-quake.time>maxDT:
            record[1]=0.0
            record[2]=0.0
    return quake


def preGan(waveformL,maxCount=10000,indexL=np.arange(-200,200)):
    realFile='gan/input/waveform4.mat'
    resFile='gan/output/genWaveform.mat'
    modelFile='gan/model/phaseGen'
    boardDir='gan/boardDir/'
    if not os.path.exists(os.path.dirname(realFile)):
        os.mkdir(os.path.dirname(realFile))
    if not os.path.exists(os.path.dirname(resFile)):
        os.mkdir(os.path.dirname(resFile))
    if not os.path.exists(os.path.dirname(modelFile)):
        os.mkdir(os.path.dirname(modelFile))
    if not os.path.exists(os.path.dirname(boardDir)):
        os.mkdir(os.path.dirname(boardDir))
    waveformAll=np.zeros((maxCount,indexL.size,1,3))
    count = 0
    for waveform in waveformL:
        count
        if count>=maxCount:
                break
        for i in range(waveform['pWaveform'].shape[0]):
            index0=np.where(waveform['indexL'][0,:]==0)[0]+int(200-100*np.random.rand(1))
            w=waveform['pWaveform'][i,indexL+index0,:]
            if min(abs(w).max(axis=1))<=0:
                print('badWaveform')
                continue
            w=w/np.linalg.norm(w)*3
            waveformAll[count,:,:,:]=w.reshape((indexL.size,1,3))
            count+=1
            if count>=maxCount:
                break
    sio.savemat(realFile,{'waveform':waveformAll[:count,:,:,:]})

def plotGan():
    realFile='gan/input/waveform4.mat'
    resFile='gan/output/genWaveform.mat'
    outDir=os.path.dirname(resFile)
    waveformO=sio.loadmat(realFile)['waveform']
    waveformGen=sio.loadmat(resFile)['genWaveform']
    timeL=np.arange(400)*0.02
    for i in range(4):
        plt.subplot(2,2,i+1)
        for comp in range(3):
            plt.plot(timeL,waveformO[i,:,0,comp]+2-comp,'b')
        plt.yticks(np.arange(3),['Z','N','E'])
        plt.suptitle('real waveforms')
    plt.savefig(outDir+'/real.png')
    plt.close()

    for i in range(4):
        plt.subplot(2,2,i+1)
        for comp in range(3):
            plt.plot(timeL,waveformGen[i,:,0,comp]+2-comp,'b')
        plt.yticks(np.arange(3),['Z','N','E'])
        plt.suptitle('fake waveforms')
    plt.savefig(outDir+'/fake.png')
    plt.close()

def findNear(time,timeL,maxD=5):
    if np.abs(timeL-time).min()<maxD:
        return np.abs(timeL-time).argmin()
    else:
        return -1

def compareQuakeL(quakeL1, quakeL2,recordF=None):
    PS1=getQuakeInfoL(quakeL1)
    PS2=getQuakeInfoL(quakeL2)
    #print(PS1[:,0])
    #print(PS2[:,0])
    h1,=plt.plot(PS1[:,2],PS1[:,1],'.b')
    h2,=plt.plot(PS2[:,2],PS2[:,1],'.r')
    for i in range(len(quakeL1)):
        index= findNear(PS1[i,0],PS2[:,0])
        if index>=0:
            laL=np.array([PS1[i,1],PS2[index,1]])
            loL=np.array([PS1[i,2],PS2[index,2]])
            hh,=plt.plot(loL,laL,'g')
            if recordF!=None:
                recordF.write("%02d %02d %.2f %7.4f %7.4f %7.4f %4.2f %4.2f\n"%(i,\
                    index,PS1[i,0],PS1[i,0]-PS2[index,0],PS1[i,1]-PS2[index,1],\
                    PS1[i,2]-PS2[index,2],quakeL1[i].cc,quakeL2[index].cc))
    return h1,h2,hh

def onlyQuake(quakeL,quakeRefL):
    qL=[]
    for quake in quakeL:
        isM=False
        for quakeRef in quakeRefL:
            if quake.tmpName==quakeRef.filename:
                isM=True
                break
        if isM:
            qL.append(quake)
    return qL

def analysisMFT(quakeL1,quakeL2,quakeRefL,filename='wlx/MFTCompare.png',recordName='tmp.res'):
    quakeL1=onlyQuake(quakeL1,quakeRefL)
    quakeL2=onlyQuake(quakeL2,quakeRefL)
    with open(recordName,'w+') as f:
        f.write("i1 i2 oTime          dTime   dLa     dLo    cc1  cc2\n")
        h1,h2,hh=compareQuakeL(quakeL1,quakeL2,recordF=f)
    PSRef=getQuakeInfoL(quakeRefL)
    hRef,=plt.plot(PSRef[:,2],PSRef[:,1],'^k') 
    plt.legend((h1,h2,hh,hRef),('Jiang','WLX','same','Ref'))
    plt.xlabel('lo')
    plt.ylabel('la')
    plt.xlim([-0.025,0.025])
    plt.ylim([-0.025,0.025])
    plt.savefig(filename,dpi=300)
    plt.close()

def analysisMFTAll(quakeL1,quakeL2,quakeRefL,outDir='wlx/compare/'):
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    for quakeRef in quakeRefL:
        figName=outDir+'/'+quakeRef.filename.split('/')[-1].split('.')[0]+'.png'
        recordName=outDir+'/'+quakeRef.filename.split('/')[-1].split('.')[0]+'.res'
        analysisMFT(quakeL1,quakeL2,[quakeRef],filename=figName,recordName=recordName)

def dTimeQuake(quake,quakeRef,staInfos,filename='test.png',quake2=None):
    ishh1=False
    for R in quake:
        for RR in quakeRef:
            if R.getStaIndex()==RR.getStaIndex() and R.pTime()!=0 and RR.pTime()!=0:
                staInfo=staInfos[R.getStaIndex()]
                AZ=(DistAz(quake.loc[0],quake.loc[1],\
                    staInfo['la'],staInfo['lo']).getAz()+180)/180*np.pi
                dTime=(R.pTime()-quake.time-(RR.pTime()-quakeRef.time))/30
                dLa=dTime*np.cos(AZ)
                dLo=dTime*np.sin(AZ)
                laL=np.array([0,dLa])+quake.loc[0]
                loL=np.array([0,dLo])+quake.loc[1]
                if R.getPCC()>0.5:
                    hh,=plt.plot(loL,laL,'k')
                else:
                    hh1,=plt.plot(loL,laL,'y')
                    ishh1=True
    h,=plt.plot(quake.loc[1],quake.loc[0],'.b')
    hR,=plt.plot(quakeRef.loc[1],quakeRef.loc[0],'^r')
    hL=(h,hR,hh)
    nameL=('jiang','Ref','dTime/30 cc>0.5')
    if ishh1:
        hL=hL+(hh1,)
        nameL=nameL+('dTime/30 cc<0.5',)
    if quake2!=None:
        h2,=plt.plot(quake2.loc[1],quake2.loc[0],'*b')
        hL=hL+(h2,)
        nameL=nameL+('WLX',)
    plt.legend(hL,nameL)
    plt.xlim([quakeRef.loc[1]-0.015,quakeRef.loc[1]+0.015])
    plt.ylim([quakeRef.loc[0]-0.015,quakeRef.loc[0]+0.015])
    plt.savefig(filename,dpi=300)
    plt.close()

def dTimeQuakeByRef(quakeL,quakeRef,staInfos,outDir='wlx/dTime/',quakeL2=None):
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    quakeL=onlyQuake(quakeL,[quakeRef])
    quakeL2=onlyQuake(quakeL2,[quakeRef])
    PS2=getQuakeInfoL(quakeL2)
    for quake in quakeL:
        figName=outDir+'/'+quake.filename.split('/')[-1].split('.')[0]+'.png'
        index= findNear(quake.time,PS2[:,0])
        if index>=0:
            dTimeQuake(quake,quakeRef,staInfos,filename=figName,quake2=quakeL2[index])

def genKY(sacDir='/home/jiangyr/hiNet/event/',delta=0.02):
    N=10000
    pxFile='PX2.mat'
    pyFile='PY2.mat'
    sxFile='SX2.mat'
    syFile='SY2.mat'
    dTimeFile='dTime.mat'
    PX=np.zeros((N,3000,3))
    PY=np.zeros((N,3000,1))
    SX=np.zeros((N,3000,3))
    SY=np.zeros((N,3000,1))
    dTime=np.zeros(N)
    pCount=0
    sCount=0
    indexO=1500
    iO=400000
    indexL=np.arange(-1500,1500)
    pY0=np.exp(-((np.arange(800000)-iO)/5)**2)
    sY0=np.exp(-((np.arange(800000)-iO)/10)**2)
    f0=int(1/delta)
    for monthDir in glob(sacDir+'2*/'):
        print(monthDir)
        for eventDir in glob(monthDir+'/D*/'):
            for sacZ in glob(eventDir+'/*U.SAC'):
                strL='ENZ'
                sacE=sacZ[:-5]+'E'+sacZ[-5+1:]
                sacN=sacZ[:-5]+'N'+sacZ[-5+1:]
                sacFileL=[sacE,sacN,sacZ]
                #print(sacFileL)
                isF=True
                for sac in sacFileL:
                    if not os.path.exists(sac):
                        isF=False
                #if os.path.exists()
                if not isF:
                    continue
                sacL=[obspy.read(sac)[0] for sac in sacFileL]
                downSampleRate=delta/sacL[0].stats['delta']
                if downSampleRate<1 or np.abs(round(downSampleRate) -downSampleRate)>0.01 :
                    continue
                downSampleRate=int(round(downSampleRate))
                bTime=sacL[0].stats['starttime'].timestamp
                eTime=sacL[0].stats['endtime'].timestamp
                pTime=bTime+sacL[0].stats['sac']['t0']-sacL[0].stats['sac']['b']
                sTime=bTime+sacL[0].stats['sac']['t1']-sacL[0].stats['sac']['b']

                oTime=pTime+(np.random.rand()-0.5)*40
                oTimeP=min(max(oTime,bTime+30.1),eTime-30.1)
                dIndexP=int(round((oTimeP-pTime)/delta))
                if pTime>bTime+1000:
                    continue
                if sTime>bTime+1000:
                    continue
                #print(sacL[0].stats['sac']['t0'],pTime,oTimeP,bTime,eTime,dIndexP)
                PY[pCount,:,0]=pY0[iO+dIndexP+indexL]

                oTime=sTime+(np.random.rand()-0.5)*40
                oTimeS=min(max(oTime,bTime+30.1),eTime-30.1)
                dIndexS=int(round((oTimeS-sTime)/delta))
                #PY[pCount,:,0]=pY0[iO+dIndexP+indexL]
                SY[sCount,:,0]=sY0[iO+dIndexS+indexL]
                try :
                    for comp in range(3):
                        sac=sacL[comp]
                        sac.decimate(downSampleRate)
                        oIndexP=int(round((oTimeP-bTime)*f0))
                        PX[pCount,:,comp]=sac.data[oIndexP+indexL]
                        oIndexS=int(round((oTimeS-bTime)*f0))
                        SX[sCount,:,comp]=sac.data[oIndexS+indexL]
                except:
                    print('wrong')
                    continue
                else:
                    pass

                dTime[pCount]=sTime-pTime
                pCount+=1
                sCount+=1
                if pCount%100==0:
                    print(pCount,sCount)
                if pCount%1000==0:
                    x=PX[pCount-1,:,0]
                    plt.plot(x/x.max())
                    plt.plot(PY[pCount-1,:,0]-1)
                    x=SX[sCount-1,:,0]
                    plt.plot(x/x.max()-2)
                    plt.plot(SY[sCount-1,:,0]-3)
                    plt.savefig('fig/%d.png'%pCount)
                    plt.close()
                if pCount==N:
                    break
            if pCount==N:
                break
        if pCount==N:
            break
    if pCount<10000:
        print('No')
        return
    h5py.File(pxFile,'w')['px']=PX[:pCount]
    h5py.File(pyFile,'w')['py']=PY[:pCount]
    h5py.File(sxFile,'w')['sx']=SX[:pCount]
    h5py.File(syFile,'w')['sy']=SY[:pCount]
    h5py.File(dTimeFile,'w')['dTime']=dTime[:pCount]


def genKYSC():
    delta=0.02
    sacDir='/home/jiangyr/WC_mon78/'
    phaseLst='phaseLst0'
    staName0=''
    fileName='SC.mat'
    date0=0
    N=15000
    PX=np.zeros((N,3000,3))
    PY=np.zeros((N,3000,1))
    SX=np.zeros((N,3000,3))
    SY=np.zeros((N,3000,1))
    pCount=0
    sCount=0
    indexO=1500
    iO=2500
    indexL=np.arange(-1500,1500)
    pY0=np.exp(-((np.arange(5000)-iO)/5)**2)
    sY0=np.exp(-((np.arange(5000)-iO)/10)**2)
    with open(phaseLst) as f:
        for line in f.readlines():
            if line[0]=='2':
                continue
            staName=line[:3]
            Y=int(line[5:9])
            M=int(line[9:11])
            D=int(line[11:13])
            h=int(line[13:15])
            m=int(line[15:17])
            s=float(line[17:21])
            phase=line[22]
            time=UTCDateTime(Y,M,D,h,m,s).timestamp-3600*8
            if staName!=staName0 or np.floor(time/86400)!=date0:
                print(staName0,time)
                #XX.MXI.2008189000000.BHZ
                timeStr=UTCDateTime(time).strftime('%Y%j')
                print('%s/*.%s*%s*Z'%(sacDir,staName,timeStr))
                tmp=glob('%s/*.%s*%s*Z'%(sacDir,staName,timeStr))
                if len(tmp)<1:
                    continue
                sacZ=tmp[0]
                sacN=sacZ[:-1]+'N'
                sacE=sacZ[:-1]+'E'
                sacFileL=[sacE,sacN,sacZ]
                isF=True
                for sac in sacFileL:
                    if not os.path.exists(sac):
                        isF=False
                #if os.path.exists()
                if not isF:
                    continue
                staName0=staName
                date0=np.floor(time/86400)
                sacL=[obspy.read(sac)[0] for sac in sacFileL]
                [sac.decimate(2) for sac in sacL]
            oTime=time+(np.random.rand()-0.5)*40
            try:
                for comp in range(3):
                    sac=sacL[comp]
                    bTime=sac.stats['starttime'].timestamp
                    #print(bTime)
                    bIndex=int(round((oTime-bTime)/delta))
                    dIndex=int(round((oTime-time)/delta))
                    #print(bIndex,dIndex)
                    if phase =='P':
                        PX[pCount,:,comp]=sac.data[bIndex+indexL]
                        PY[pCount,:,0]=pY0[iO+dIndex+indexL]
                    else:
                        SX[sCount,:,comp]=sac.data[bIndex+indexL]
                        SY[sCount,:,0]=sY0[iO+dIndex+indexL]
            except:
                print('wrong')
                continue
            else:
                pass
            if phase=='P':
                if ((PX[pCount,:,:]**2).sum(axis=0)==0).sum()>0:
                    print('wrong data')
                    continue
                pCount+=1
            else:
                if ((SX[sCount,:,:]**2).sum(axis=0)==0).sum()>0:
                    print('wrong data')
                    continue
                sCount+=1

            if pCount%100==0:
                    print(pCount,sCount)
                    print((PX[max(pCount-1,0),:,:]**2).sum(axis=0))
            if pCount%1000==0:
                x=PX[pCount-1,:,0]
                plt.plot(x/x.max())
                plt.plot(PY[pCount-1,:,0]-1)
                x=SX[sCount-1,:,0]
                plt.plot(x/x.max()-2)
                plt.plot(SY[sCount-1,:,0]-3)
                plt.savefig('fig/SC_%d.png'%pCount)
                plt.close()
            if pCount==N:
                break
        sio.savemat(fileName,{'px':PX[:pCount],'py':PY[:pCount],'sx':SX[:sCount],'sY':SY[:sCount]})

def fileRes(fileName,phase='p'):
    data=sio.loadmat(fileName)
    y0JP=data['%sy0'%phase]
    yJP=data['out%sy'%phase]
    y0SC=data['%sy0Test'%phase]
    ySC=data['out%syTest'%phase]
    return cmpY(y0JP,yJP,phase=phase),cmpY(y0SC,ySC,phase=phase)


def cmpY(y0,y,delta=0.02,phase='p'):
    if phase=='p':
        i0=250
        i1=1750
    else:
        i0=500
        i1=1500
    y0=y0.reshape(-1,y0.shape[1])
    y=y.reshape(-1,y.shape[1])
    index0=y0.argmax(axis=1)
    v0=y0[:,i0:i1].max(axis=1)
    index=y[:,i0:i1].argmax(axis=1)+i0
    v=y[:,i0:i1].max(axis=1)
    t0=index0*delta
    t0[v0<0.99]=-100000
    t=index*delta
    V0=v0
    V=v
    return t0,V0,t,V

def calRes(t0,V0,t,V,minDT=0.5):
    dT=t-t0
    m=dT[np.abs(dT)<2].mean()
    dT=dT-dT[np.abs(dT)<2].mean()
    Tp=((np.abs(dT)<=minDT) * (V>0.5)*(V0>0.99)).sum()
    Fp=((np.abs(dT)>minDT) * (V>=0.5)*(V0>0.99)).sum()
    #( (V>0.5)*(V0<0.0001)).sum()
    Fn=((V<0.5) * (V0>0.5)).sum()+((V>=0.5) * (V0>0.5)*(np.abs(dT)>minDT)).sum()
    p=Tp/(Tp+Fp)
    r=Tp/(V0>0.99).sum()
    F1=2*p*r/(p+r)
    dTNew=dT[(np.abs(dT)<=minDT) * (V>0.5)]
    return p,r,F1,dTNew.mean()+m,dTNew.std()

def calResAll():
    minDTL=[1,0.5,0.25]
    mulL=[128,64,32,16,4,1]
    pFileL=['resDataP_80000_990-2-15']+['resDataP_%d_0-2-15'%(320000/mulL[i]) for i in range(6)]+['resDataP_320000_100-2-15']#+['resDataP_320000_100']
    sFileL=['resDataS_80000_990-2-15']+['resDataS_%d_0-2-15'%(320000/mulL[i]) for i in range(6)]+['resDataS_320000_100-2-15']#+['resDataS_320000_100']
    cmdStrL=['b','-.g','g','-.r','r','-.k','k','y']
    strL=['SC']+['1/%d JP'%(mulL[i]) for i in range(6)]+['SC + 1/1 JP']#+['no Filter']
    timeBinL=np.arange(-1.5,1.51,0.1)
    pJPL=[]
    sJPL=[]
    pSCL=[]
    sSCL=[]
    sL='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    resPJP='resPJP-2-15'
    resPSC='resPSC-2-15'
    resSJP='resSJP-2-15'
    resSSC='resSSC-2-15'
    excelName='resPSJPSC.xlsx'
    for pFile in pFileL:
        pJP,pSC=fileRes(pFile,'p')
        pJPL.append(pJP)
        pSCL.append(pSC)

    for sFile in sFileL:
        sJP,sSC=fileRes(sFile,'s')
        sJPL.append(sJP)
        sSCL.append(sSC)
    textL='abcd'
    psL=[pJPL,pSCL,sJPL,sSCL]
    fileL=[resPJP,resPSC,resSJP,resSSC]
    plt.close()
    plt.figure(figsize=[10,10])
    excel=Workbook()
    for index in range(4):
        ps=psL[index]
        fileName=fileL[index]
        sheet=excel.create_sheet(fileName, index=index)
        plt.subplot(2,2,index+1)
        plt.xlim([-1.5,1.5])
        plt.ylim([0,0.6])
        plt.text(-1.42,0.57,'(%s)'%textL[index])
        hL=()
        for i in range(len(ps)):
            tmp=ps[i]
            tmpStr=strL[i]
            cmdStr=cmdStrL[i]
            dTime=(tmp[2]-tmp[0])[(tmp[1]>0.5) * (tmp[3]>0.5)]
            tmpFL=np.histogram(dTime,timeBinL,normed=True)[0]*(timeBinL[1]-timeBinL[0])
            h,=plt.plot((timeBinL[:-1]+timeBinL[1:])/2,tmpFL,cmdStr,linewidth=0.5)
            hL+=(h,)
            #f.write('%11s : '%tmpStr)
            row=[]
            for minDT in minDTL:
                p,r,F1,m,s=calRes(tmp[0],tmp[1],tmp[2],tmp[3],minDT)
                #f.write(' %5.3f %5.3f %5.3f %5.3f %5.3f |'%(p,r,F1,m,s))
                row+=[p,r,F1,m,s]
            #f.write('\n')
            sheet.append(row)
            plt.legend(hL,strL)
    excel.save(excelName)
    plt.savefig('NM/trainChange.eps')
    plt.close()

def processX(X, rmean=True, normlize=True, reshape=True):
    if reshape:
        X = X.reshape(-1, 2000, 1, 3)
    if rmean:
        X = X - X.mean(axis=(1, 2)).reshape([-1, 1, 1, 3])
    if normlize:
        X = X/(X.std(axis=(1, 2, 3)).reshape([-1, 1, 1, 1]))
    return X

def plotWaveform(x0,y0,y,delta=0.02,figName='test.eps',phase='p',text='(a)'):
    timeL=np.arange(x0.shape[0])*0.02
    x0=processX(x0)
    x0=x0/np.abs(x0).max(axis=(1,2,3),keepdims=True)*0.4
    y0=y0.reshape((-1))
    if phase=='p':
        y0=y0**0.25
    y=y.reshape((-1))
    t0=(y0[250:1750].argmax()+250)*delta
    t=(y[250:1750].argmax()+250)*delta
    plt.figure(figsize=[4,4])
    for comp in range(3):
        plt.plot(timeL,x0[0,:,0,comp]+comp,'k',linewidth=0.5)
    plt.plot(timeL,y0-1.5,'-.b',linewidth=0.5)
    plt.xlim([timeL[0],timeL[-1]])
    plt.yticks(np.arange(-2,3),['q(x)','p(x)','E','N','Z'])
    plt.xlabel('t/s')
    h0,=plt.plot(np.array([t0,t0]),np.array([-1.2,-0.4]),'b',linewidth=0.5)
    plt.legend((h0,),{'t0'})
    plt.ylim([-1.6,3.2])
    plt.savefig(figName[:-4]+'_0.eps')
    h1,=plt.plot(np.array([t,t]),np.array([-2.2,-1.4]),'r',linewidth=0.5)
    plt.ylim([-2.6,3.2])
    plt.plot(timeL,y-2.5,'-.r',linewidth=0.5)
    plt.legend((h0,h1),{'t0','t'})
    plt.text(1,2.95,text)
    plt.savefig(figName)
    plt.close()

def plotTestOutput(fileName='resDataP_320000_100-2-15',phase='p',outDir='NM/testFig/',N=100):
    data=sio.loadmat(fileName)
    y0JP=data['%sy0'%phase]
    x0JP=data['out%sx'%phase]
    yJP=data['out%sy'%phase]
    y0SC=data['%sy0Test'%phase]
    x0SC=data['out%sxTest'%phase]
    ySC=data['out%syTest'%phase]
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    for i in range(200):
        if phase=='p':
            strTmp='(a)'
        else:
            strTmp='(c)'
        plotWaveform(x0JP[i],y0JP[i],yJP[i],figName='%s/JP%d.eps'%(outDir,i),phase=phase,text=strTmp)
    for i in range(200):
        if phase=='p':
            strTmp='(b)'
        else:
            strTmp='(d)'
        plotWaveform(x0SC[i],y0SC[i],ySC[i],figName='%s/SC%d.eps'%(outDir,i),phase=phase,text=strTmp)


def plotWaveformByMat(quake,staInfos,matDir='NM/output20190901/',mul=0.15,compP=2,compS=2,outDir='NM/output20190901V2/'):
    #loadWaveformByQuake(quake,matDir='output',isCut=False,index0=-250,index1=250,f=[-1,-1]):
    waveform=loadWaveformByQuake(quake,matDir=matDir)
    pWaveform=waveform['pWaveform'].reshape((-1,waveform['pWaveform'].shape[1],1,3))
    sWaveform=waveform['sWaveform'].reshape((-1,waveform['sWaveform'].shape[1],1,3))
    pWaveform/=pWaveform.max(axis=1,keepdims=True)/mul
    sWaveform/=sWaveform.max(axis=1,keepdims=True)/mul
    staIndexL=waveform['staIndexL'][0].astype(np.int64)

    eqLa=quake.loc[0]
    eqLo=quake.loc[1]
    maxDis=0
    minDis=100
    pTimeL=quake.getPTimeL(staInfos)
    sTimeL=quake.getPTimeL(staInfos)

    for i in range(len(staIndexL)):
        staInfo=staInfos[staIndexL[i]]
        timeL=waveform['indexL'][0]*waveform['deltaL'][0][i]
        dis=DistAz(eqLa,eqLo,staInfo['la'],staInfo['lo']).getDelta()
        maxDis=max(dis,maxDis)
        minDis=min(dis,minDis)

        if pTimeL[staIndexL[i]]!=0 and waveform['pTimeL'][0][i]!=0:
            plt.subplot(2,1,1)
            plt.plot(timeL+ 0*(waveform['pTimeL'][0][i]-quake.time),pWaveform[i,:,0,compP]+dis,'k',linewidth=0.5)
            oTime=(timeL[0]+timeL[-1])/2
            #plt.plot(np.array([oTime,oTime])+ waveform['pTimeL'][0][i]-quake.time,np.array([dis-0.5,dis+0.5]),'-.r',linewidth=0.5)
        if sTimeL[staIndexL[i]]!=0 and waveform['sTimeL'][0][i]!=0:
            plt.subplot(2,1,2)
            plt.plot(timeL+ 0*(waveform['sTimeL'][0][i]-quake.time),sWaveform[i,:,0,compS]+dis,'k',linewidth=0.5)
            oTime=(timeL[0]+timeL[-1])/2
            #plt.plot(np.array([oTime,oTime])+ waveform['sTimeL'][0][i]-quake.time,np.array([dis-0.5,dis+0.5]),'-.r',linewidth=0.5)
    for i in range(2):
        plt.subplot(2,1,i+1)
        plt.xlim([timeL[0],timeL[-1]])
        h0,=plt.plot(np.array([oTime,oTime])*0,np.array([minDis-2.5,maxDis+2.5]),'-.k',linewidth=1)
        if i==0:
            plt.legend((h0,),['p'])
        else:
            plt.legend((h0,),['s'])
        plt.ylim(minDis-0.5,maxDis+0.8)
        if i==1:
            plt.xlabel('t/s')
        plt.ylabel('D/Rad')
    dirName=os.path.dirname('%s/%s.eps'%(outDir,quake.filename[:-4]))
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    plt.savefig('%s/%s.eps'%(outDir,quake.filename[:-4]))
    plt.savefig('%s/%s.png'%(outDir,quake.filename[:-4]),dpi=300)
    plt.close()

def dayTimeDis(quakeLs,staInfos,mlL0,minCover=0.5,minSta=3,isBox=False):
    #timeL=[]
    depL=[]
    mlL=[]
    numL=[]
    numLS=[]
    timeL=[]
    laL=[]
    loL=[]
    plt.close()
    plt.figure(figsize=[12,4])
    for quakeL in quakeLs:
        for quake in quakeL:
            if len(quake)<minSta or quake.calCover(staInfos)<minCover:
                continue
            la=quake.loc[0]
            lo=quake.loc[1]
            if isBox and (la<38.7 or la>42.2 or lo<97.5 or lo>103.8):
                continue
            timeL.append(quake.time)
            mlL.append(quake.ml)
            depL.append(quake.loc[2])
            laL.append(quake.loc[0])
            loL.append(quake.loc[1])
            numL.append(len(quake))
            numLS.append(np.sign(quake.getSTimeL(staInfos)).sum())
    depL=np.array(depL)
    mlL=np.array(mlL)
    numL=np.array(numL)
    numLS=np.array(numLS)
    plt.subplot(1,3,1)
    plt.hist(mlL,np.arange(-1,6,0.2),color='k',log=True)
    #plt.hist(mlL0,np.arange(-1,6,0.2),color='r',log=True,)
    #plt.legend((h2,h1),['catalog','auto pick'])
    plt.xlabel('ml')
    plt.ylabel('count')
    a=plt.ylim()
    b=plt.xlim()
    plt.text(-1.1,a[1]*0.7,'(a)')

    plt.subplot(1,3,2)
    #plt.hist(mlL,np.arange(-1,6,0.2),color='b',log=True)
    plt.hist(mlL0,np.arange(-1,6,0.2),color='k',log=True,)
    #plt.legend((h2,h1),['catalog','auto pick'])
    plt.xlabel('ml')
    plt.ylabel('count')
    a=plt.ylim(a)
    b=plt.xlim(b)
    plt.text(-1.1,a[1]*0.7,'(b)')
    plt.subplot(1,3,3)
    plt.hist(numL,np.arange(0,100,1),color='k',log=True)
    plt.xlabel('n')
    a=plt.ylim(a)
    plt.text(-1.5,a[1]*0.7,'(c)')
    #plt.ylabel('count')
    plt.savefig('NM/ml_n.eps')
    plt.savefig('NM/ml_n.tiff',dpi=600)
    plt.close()
    print(len(numL),numL.sum(),numLS.sum())
    return np.array(timeL),np.array(laL),np.array(loL)

def getCatalog(fileName='NM/catalog.txt'):
    timeL=[]
    mlL=[]
    laL=[]
    loL=[]
    laL0=[]
    loL0=[]
    with open(fileName) as f:
        for line in f.readlines():
            la=float(line[24:30])
            lo=float(line[32:39])
            #print(la,lo)
            time=UTCDateTime(line[:22]).timestamp-3600*8
            if time<UTCDateTime(2015,1,1).timestamp and time>=UTCDateTime(2014,1,1).timestamp:
                if la<37.75 or la>40.7 or lo<96.2 or lo>104.2:
                    continue
                laL.append(la)
                loL.append(lo)
                mlL.append(float(line[45:49]))
                if la<38.7 or la>42.2 or lo<97.5 or lo>103.8:
                    continue
                timeL.append(time)
                laL0.append(la)
                loL0.append(lo)
                
    return np.array(timeL),np.array(mlL),np.array(laL),np.array(loL),np.array(laL0),np.array(loL0)

def compareTime(timeL,timeL0,laL,loL,laL0,loL0,maxDT=10,maxD=2):
    count=0
    for i0 in range(len(timeL0)):
        time0=timeL0[i0]
        if np.abs(timeL-time0).min()<maxDT:
            i=np.abs(timeL-time0).argmin()
            if np.abs(laL[i]-laL0[i0])<maxD and np.abs(loL[i]-loL0[i0])<maxD:
                count+=1
    return count
