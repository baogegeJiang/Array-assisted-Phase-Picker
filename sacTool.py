import obspy
from obspy import UTCDateTime
from distaz import DistAz
import numpy as np
from scipy import interpolate as interp
import obspy.core.trace as trace
import obspy.core.stream as stream
import os
import tool
from obspy.taup import TauPyModel
import matplotlib.pyplot as plt
import time


def getTimeLim(sacs):
    n = len(sacs)
    bTime = UTCDateTime(1970,1,1)
    eTime = UTCDateTime(2200,12,31)
    for i in range(n):
        bTime = max([sacs[i].stats.starttime, bTime])
        eTime = min([sacs[i].stats.endtime, eTime])
    return bTime, eTime


class data:
    def getDataByTimeL(self, timeL, kind="slinear", fill_value='extrapolate'):
        timeL0=self.getTimeL()
        N=min(len(timeL0), self.data.shape[0])
        timeL0=timeL0[range(N)]
        if len(self.data.shape) > 1:
            return interp.interp1d(timeL0, self.data[range(N), :], kind=kind, \
                fill_value=fill_value, axis=0)(timeL)
        else:
            return interp.interp1d(timeL0, self.data[range(N)], kind=kind, \
                fill_value=fill_value, axis=0)(timeL)
        


class Data(data):
    def __init__(self, data, bTime=-999, eTime=-999, delta=0, freq=[-1, -1]):
        self.data = data
        self.convert2Float32()
        self.bTime = bTime
        self.eTime = eTime
        self.delta = delta
        self.freq = freq
    def getTimeL(self):
        return np.arange(self.bTime.timestamp, self.eTime.timestamp, \
            self.delta)
    def getDataByTimeLQuick(self, timeL):
        if timeL.size<2:
            return self.getDataByTimeL(timeL)
        if round(self.delta*100000) != round((timeL[1]-timeL[0])*100000):
            return self.getDataByTimeL(timeL)
        bTime = timeL[0]
        bTime0 = self.bTime.timestamp
        bIndex = int((bTime-bTime0)/self.delta)
        eIndex = bIndex + len(timeL)
        return self.data[bIndex:eIndex,:]
    def convert2Float32(self):
        self.data=self.data.astype(np.float32)


class sac(trace.Trace, data):
    def __init__(self, object):
        self.stats=object.stats
        self.data=object.data
    def getTimeL(self):
        bTime = self.stats.starttime
        eTime = self.stats.endtime
        timeL = np.arange(bTime.timestamp, eTime.timestamp, self.stats.delta)
        return timeL
    def getDataByTimeLQuick(self, timeL):
        f0 = round(1/(timeL[1]-timeL[0]))
        bTime = timeL[0]
        n = len(timeL)
        #return self.data[:]
        tmpL=self.split()
        try:
            if len(tmpL)>1:
                ID=tmpL[0].id
                for tmp in tmpL:
                    #print(tmp)
                    tmp.id=ID
                self=sac(tmpL.merge(fill_value=0)[0])
            print(self)
            self.interpolate(f0, starttime=bTime, npts=n)
        except:
            print('wrong interp')
            return timeL*0
        else:
            return self.data[:]

        
def sac2data(sacs, delta0=0.02):
    bTime, eTime = getTimeLim(sacs)
    timeL = np.arange(bTime.timestamp, eTime.timestamp, delta0)
    data = np.zeros((timeL.shape[0], len(sacs)))
    for i in range(len(sacs)):
        data[:, i] = sac(sacs[i]).getDataByTimeLQuick(timeL)
    return data, bTime, eTime, delta0


def mergeSacByName(sacFileNames, delta0=0.02, freq=[-1, -1], \
    filterName='bandpass', corners=2, zerophase=True,maxA=1e5):
    count = 0
    sacM = None
    tmpSacL=None
    for sacFileName in sacFileNames:
        try:
            #time0=time.time()
            tmpSacs=obspy.read(sacFileName, debug_headers=True,dtype=np.float32)
            #time1=time.time()
            if freq[0] > 0:
                tmpSacs.detrend('demean').detrend('linear').filter(filterName,\
                        freqmin=freq[0], freqmax=freq[1], \
                        corners=2, zerophase=zerophase)
            else:
                tmpSacs.detrend('demean').detrend('linear')
            #time2=time.time()

        except:
            print('wrong read sac')
            continue
        else:
            if tmpSacL==None:
                tmpSacL=tmpSacs
            else:
                tmpSacL+=tmpSacs
    if tmpSacL!=None and len(tmpSacL)>0:
        ID=tmpSacL[0].id
        for tmp in tmpSacL:
            try:
                tmp.id=ID
            except:
                pass
            else:
                pass
        try:
            sacM=tmpSacL.merge(fill_value=0)[0]
            std=sacM.std()
            if std>maxA:
                print('#####too many noise std : %f#####'%std)
                sacM=None
            else:
                print('#####min noise std : %f#####'%std)
        except:
            print('wrong merge')
            sacM=None
        else:
            print(sacM)
        #time3=time.time()
        #print('readSac',time1-time0,'filter',time2-time1,'merge',time3-time2)
    return sacM

def checkSacFile(sacFileNamesL):
    if len(sacFileNamesL)==0:
        return False
    for sacFileNames in sacFileNamesL:
        count = 0
        for sacFileName in sacFileNames:
            if os.path.exists(sacFileName):
                count=count+1
        if count==0:
            return False
    return True

def getDataByFileName(sacFileNamesL, delta0=0.02, freq=[-1, -1], \
    filterName='bandpass', corners=4, zerophase=True,maxA=1e5):
    if not checkSacFile(sacFileNamesL):
        return Data(np.zeros(0), -999, -999, 0, [-1 -1])
    sacs = stream.Stream()
    #time0=time.time()
    for sacFileNames in sacFileNamesL:
        tmp=mergeSacByName(sacFileNames, delta0=delta0,freq=freq,\
            filterName=filterName,corners=corners,zerophase=zerophase,maxA=maxA)
        if tmp == None:
            print('False')
            return Data(np.zeros(0), -999, -999, 0, [-1 -1])
        else:
            sacs.append(tmp)
    time1=time.time()
    dataL=sac2data(sacs, delta0=delta0)
    time2=time.time()
    #print('read',time1-time0,'dec',time2-time1)
    return Data(dataL[0], dataL[1], dataL[2], dataL[3], freq)

class subarea:
    def __init__(self, minLa, minLo, maxLa, maxLo, midLa, midLo):
        self.minLa=minLa
        self.minLo=minLo
        self.maxLa=maxLa
        self.maxLo=maxLo
        self.midLa=midLa
        self.midLo=midLo
        self.R = self.getR()
    def getR(self):
        res=DistAz(self.minLa, self.minLo, self.maxLa, self.maxLo)
        return res.delta/2

class areaMat:
    def __init__(self, laL, loL, laN, loN):
        subLaL = tool.divideRange(laL, laN)
        subLoL = tool.divideRange(loL, loN)
        self.laN = laN
        self.loN = loN
        self.subareas = [[subarea(subLaL['minL'][i],subLoL['minL'][j],\
            subLaL['maxL'][i], subLoL['maxL'][j],\
            subLaL['midL'][i], subLoL['midL'][j]\
            ) for j in range(loN)] for i in range(laN)]

    def __iter__(self):
        return list(self.subareas).__iter__()

    def __setitem__(self, index, trace):
        self.subareas.__setitem__(index, trace)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(traces=self.subareas.__getitem__(index))
        else:
            return self.subareas.__getitem__(index)

    def __delitem__(self, index):
        return self.subareas.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        return self.__class__(traces=self.subareas[max(0, i):max(0, j):k])
        
class staTimeMat:
    def __init__(self, loc, aMat, taupM=TauPyModel(model="iasp91")):
        self.loc=loc
        minTimeP, maxTimeP = self.getMatTime(aMat, taupM=taupM, \
             phaseL=['p', 'P', 'Pn'])
        minTimeS, maxTimeS = self.getMatTime(aMat, taupM=taupM, \
             phaseL=['s', 'S', 'Sn'])
        self.minTimeP = minTimeP
        self.minTimeS = minTimeS
        self.maxTimeP = maxTimeP
        self.maxTimeS = maxTimeS
        self.minTimeD = minTimeS-minTimeP
        self.maxTimeD = maxTimeS-maxTimeP

    def getMatTime(self, aMat, taupM=TauPyModel(model="iasp91"), \
     phaseL=['P']):
        minTime = np.zeros((aMat.laN, aMat.loN))
        maxTime = np.zeros((aMat.laN, aMat.loN))
        for i in range(aMat.laN):
            for j in range(aMat.loN):
                dis=DistAz(self.loc[0], self.loc[1], aMat[i][j].midLa, \
                    aMat[i][j].midLo)
                minDis=max(0, dis.delta-aMat[i][j].R)
                maxDis=dis.delta+ aMat[i][j].R
                if minDis==0:
                    minTime[i][j]=0
                else:
                     minTime[i][j] = self.getEarist(taupM.get_travel_times(10, minDis, \
                        phase_list=phaseL))
                maxTime[i][j] = self.getEarist(taupM.get_travel_times(0, maxDis, \
                    phase_list=phaseL))
                maxTime[i][j] = max(maxTime[i][j], self.getEarist(taupM.get_travel_times(20, maxDis, \
                    phase_list=phaseL)))
                maxTime[i][j] = max(maxTime[i][j], self.getEarist(taupM.get_travel_times(60, maxDis, \
                    phase_list=phaseL)))
        return minTime, maxTime

    def getEarist(self, arrivals):
        time=10000
        for arrival in arrivals:
            time = min(time, arrival.time)
        return time



def cutSacByQuake(quake,staInfos,getFilename,comp=['BHE','BHN','BHZ'],\
    R=[-90,90,-180,180],outDir='SACCUT/',delta=0.01):
    time=quake.time
    YmdHMSj0=tool.getYmdHMSj(UTCDateTime(time))
    YmdHMSj1=tool.getYmdHMSj(UTCDateTime(time+2*3600+10))
    tmpDir=outDir+str(int(time))+'/'
    if not os.path.exists(tmpDir):
        os.mkdir(tmpDir)
    n=3600*2/delta
    for staInfo in staInfos:
        if staInfo['la']>=R[0] and \
            staInfo['la']<=R[1] and \
            staInfo['lo']>=R[2] and \
            staInfo['lo']<=R[3]:
            print(staInfo['net']+staInfo['sta'])
            for c in comp:
                sacName=staInfo['net']+staInfo['sta']+c+'.SAC'
                fileNames=getFilename(staInfo['net'],staInfo['sta'],c,YmdHMSj0)\
                +getFilename(staInfo['net'],staInfo['sta'],c,YmdHMSj1)
                fileNames=list(set(fileNames))
                sacM=mergeSacByName(fileNames,delta0=delta)
                print(sacM)
                if not sacM == None:
                    try:
                        sacM.interpolate(int(1/delta), starttime=time, npts=n)
                    except:
                        print('no data')
                        continue
                    else:
                        pass
                    sacM.write(tmpDir+sacName,format='SAC')

tmpSac1='1test.sac'
def adjust(data,loc=None,kzTime=None,tmpFile='test.sac',decMul=10,eloc=None,chn=None,sta=None,\
    net=None,o=None):
    data.decimate(decMul)
    if data.stats['_format']!='SAC':
        data.write(tmpFile,format='SAC')
        data=obspy.read(tmpFile)[0]
        #print(data)
    if loc!=None:
        data.stats['sac']['stla']=loc[0]
        data.stats['sac']['stlo']=loc[1]
        data.stats['sac']['stel']=loc[2]
    if eloc!=None:
        data.stats['sac']['evla']=eloc[0]
        data.stats['sac']['evlo']=eloc[1]
        data.stats['sac']['evdp']=eloc[2]
        dis=DistAz(eloc[0],eloc[1],loc[0],loc[1])
        dist=dis.degreesToKilometers(dis.getDelta())
        data.stats['sac']['dist']=dist
        data.stats['sac']['az']=dis.getAz()
        data.stats['sac']['baz']=dis.getBaz()
        data.stats['sac']['gcarc']=dis.getDelta()
    if chn!=None:
        data.stats['sac']['kcmpnm']=chn
        data.stats['channel']=chn
    if sta!=None and net !=None:
        data.stats['sac']['kstnm']=sta
        data.stats['station']=sta
    if o!=None:
        data.stats['sac']['o']=o
    if kzTime!=None:
        kzTime=UTCDateTime(kzTime)
        dTime=kzTime.timestamp-(data.stats.starttime.timestamp-data.stats['sac']['b'])
        data.stats['sac']['nzyear']=int(kzTime.year)
        data.stats['sac']['nzjday']=int(kzTime.julday)
        data.stats['sac']['nzhour']=int(kzTime.hour)
        data.stats['sac']['nzmin']=int(kzTime.minute)
        data.stats['sac']['nzsec']=int(kzTime.second)
        data.stats['sac']['nzmsec']=int(kzTime.microsecond/1000)
        data.stats['sac']['b']-=dTime
        #data.stats['sac']['b']=(int(data.stats['sac']['b']/data.stats.delta)*data.stats.delta)
        data.stats['sac']['e']=data.stats['sac']['b']+(data.data.size-1)*data.stats.delta
        data.write(tmpSac1)
        data=obspy.read(tmpSac1)[0]
        print(data.stats['sac'].b)
    return data

taup=[]#tool.quickTaupModel()

def cutSacByQuakeForCmpAz(quake,staInfos,getFilename,comp=['BHE','BHN','BHZ'],\
    R=[-90,90,-180,180],outDir='/home/jiangyr/cmpaz/cmpAZ/example/',delta=0.01\
    ,B=-200,E=1800,isFromO=False,decMul=10,nameMode='cmpAz',maxDT=100000):
    time0=quake.time
    tmpDir=outDir
    if not os.path.exists(tmpDir):
        os.mkdir(tmpDir)
    n=int((E-B)/delta)
    pTimeL=quake.getPTimeL(staInfos)
    for ii  in range(len(staInfos)):
        staInfo=staInfos[ii]
        if nameMode=='ML' and len(quake)>0 and (pTimeL[ii]<=0 or (pTimeL[ii]-quake.time)>maxDT):
            print('skip')
            continue 

        if staInfo['la']>=R[0] and \
            staInfo['la']<=R[1] and \
            staInfo['lo']>=R[2] and \
            staInfo['lo']<=R[3]:
            print(staInfo['net']+staInfo['sta'])
            if nameMode=='cmpAz':
                staDir=outDir+'/'+staInfo['net']+'.'+staInfo['sta']+'/'
                if not os.path.exists(staDir):
                    os.mkdir(staDir)
                rawDir=staDir+'/'+'raw/'
            bTime=time0+B
            eTime=time0+E
            YmdHMSj0=tool.getYmdHMSj(UTCDateTime(bTime))
            YmdHMSj1=tool.getYmdHMSj(UTCDateTime(eTime))
            Y=tool.getYmdHMSj(UTCDateTime(time0))
            if nameMode=='ML':
                rawDir=outDir+'/'+Y['Y']+Y['m']+Y['d']+Y['H']+Y['M']+'%05.2f/'%(time0%60)
            if not os.path.exists(rawDir):
                os.mkdir(rawDir)
            for c in comp:
                if nameMode=='cmpAz':
                    sacName=Y['Y']+Y['m']+Y['d']+'_'+Y['H']+Y['M']+'.'\
                    +staInfo['sta']+'.'+c
                if nameMode=='ML':
                    sacName=staInfo['sta']+'.'+c
                fileNames=getFilename(staInfo['net'],staInfo['sta'],c,YmdHMSj0)\
                +getFilename(staInfo['net'],staInfo['sta'],c,YmdHMSj1)
                fileNames=list(set(fileNames))
                sacM=mergeSacByName(fileNames,delta0=delta)
                print(sacM)
                if not sacM == None:
                    try:
                        sacM.interpolate(int(1/delta), starttime=bTime, npts=n)
                    except:
                        print('no data')
                        continue
                    else:
                        pass
                    if isFromO:
                        time0=bTime
                    adjust(sacM,loc=[staInfo['la'],staInfo['lo'],staInfo['dep']],kzTime=time0,\
                        decMul=decMul,eloc=quake.loc,net=staInfo['net'],sta=staInfo['sta'],\
                        chn=c)
                    os.system('mv %s  %s' % (tmpSac1,rawDir+sacName))

def cutSacByDate(date,staInfos,getFilename,comp=['BHE','BHN','BHZ'],\
    R=[-90,90,-180,180],outDir='/home/jiangyr/cmpaz/cmpAZ/example/',delta=0.01\
    ,B=-200,E=1800,isFromO=False,decMul=10,nameMode='ML'):
    time0=date.timestamp
    tmpDir=outDir
    if not os.path.exists(tmpDir):
        os.mkdir(tmpDir)
    n=int((E-B)/delta)
    #print(n)
    for staInfo in staInfos:
        if staInfo['la']>=R[0] and \
            staInfo['la']<=R[1] and \
            staInfo['lo']>=R[2] and \
            staInfo['lo']<=R[3]:
            print(staInfo['net']+'.'+staInfo['sta'])
            Y=tool.getYmdHMSj(UTCDateTime(time0))
            if nameMode=='ML':
                rawDir=outDir+'/'+Y['Y']+Y['m']+Y['d']+'/'
            if not os.path.exists(rawDir):
                os.mkdir(rawDir)
            for c in comp:
                if nameMode=='ML':
                    sacName=staInfo['sta']+'.'+c
                fileNames=getFilename(staInfo['net'],staInfo['sta'],c,Y)
                fileNames=list(set(fileNames))
                sacM=mergeSacByName(fileNames,delta0=delta)
                print(sacM)
                if not sacM == None:
                    try:
                        sacM.interpolate(int(1/delta), starttime=time0+B)
                    except:
                        print('no data')
                        continue
                    else:
                        pass
                    if isFromO:
                        time0=time0
                    adjust(sacM,loc=[staInfo['la'],staInfo['lo'],staInfo['dep']],kzTime=time0,\
                        decMul=decMul,net=staInfo['net'],sta=staInfo['sta'],\
                        chn=c,o=0)
                    os.system('mv %s  %s' % (tmpSac1,rawDir+sacName))
def preCmpAzLog(quakeL,file='/home/jiangyr/cmpaz/cmpAZ/example/eventLst'):
    with open(file,'w+') as f:
        for quake in quakeL:
            print(os.path.basename(quake.filename))
            YmdHMSj=tool.getYmdHMSj(UTCDateTime(quake.time))
            Y=YmdHMSj
            line=Y['Y']+' '+Y['m']+' '+Y['d']+' '+Y['H']+' '+Y['M']+' '+Y['S']\
            +' '+str(quake.loc[0])+' '+str(quake.loc[1])+' '+str(quake.loc[2])+\
            ' '+str(quake.ml)+' hima '+Y['Y']+Y['m']+Y['d']+'_'+Y['H']+Y['M']+'\n'
            f.write(line)
def runCmpAz(staInfos,wkDir='/home/jiangyr/cmpaz/cmpAZ/example/'):
    csh=wkDir+'cmpAZ_cshNew'
    for staInfo in staInfos:
        netsta=staInfo['net']+'.'+staInfo['sta']
        cmd='csh %s %s'%(csh,netsta)
        os.system(cmd)

def readCmpAzResult(staInfos,wkDir='/home/jiangyr/cmpaz/cmpAZ/example/',outDir='cmpAzRes/',\
    time0=UTCDateTime(2014,1,1).timestamp):
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    resL=[{} for staInfo in staInfos]
    resFile=outDir+'azLst'
    with open(resFile,'w+') as f:
        for i in range(len(staInfos)):
            staInfo=staInfos[i]
            staDir=wkDir+'/'+staInfo['net']+'.'+staInfo['sta']+'/wk/bp05_50s/'
            staRes=staDir+'cmpAZ.out.SN5'
            if not os.path.exists(staRes):
                continue
            azL=[]
            timeL=[]
            az=-999
            num=-999
            with open(staRes,'r') as sta:
                staLines=sta.readlines()
            for line in staLines:
                #print(line[:7])
                if line[:7]=='STN_HED':
                    tmp=line.split()
                    time=(UTCDateTime(tmp[1][:13]).timestamp-time0)/86400
                    timeL.append(time)
                    #print(time)
                    azL.append(float(tmp[-1]))
            line=staLines[-1]
            tmp=line.split()
            num=int(tmp[2])
            az=float(tmp[6])
            maxE=float(tmp[10])
            resL[i]['timeL']=np.array(timeL)
            resL[i]['azL']=np.array(azL)
            resL[i]['az']=az
            resL[i]['num']=num
            resL[i]['maxE']=maxE
            f.write('%s %s %d %.2f %.2f %.2f\n'%(staInfo['net'],staInfo['sta'],num,az,resL[i]['azL'].std(),maxE))
            figname=outDir+'/'+staInfo['net']+'.'+staInfo['sta']+'.png'
            plt.figure(num=1,dpi=300,clear=True)
            plt.plot(resL[i]['timeL'],resL[i]['azL'],'.b')
            plt.title('%s %s num: %d cmpAz: %.2f %.2f %.2f'%(staInfo['net'],staInfo['sta'],num,az,resL[i]['azL'].std(),maxE))
            plt.plot(resL[i]['timeL'],resL[i]['azL']*0+az,'-.r')
            plt.ylim([-180,180])
            plt.savefig(figname)
            #plt.close()

def loadCmpAz(file='cmpAzRes/azLst'):
    with open(file,'r') as f:
        lines= f.readlines()
    cmpAz={}
    for line in lines:
        t=line.split()
        net=t[0]
        sta=t[1]
        num=t[2]
        az=t[3]
        std=t[4]
        e=t[5]
        cmpAz[net+'.'+sta]=[float(az),float(e)]
    return cmpAz

def readStaInfos(fileName,cmpAz=[]):#loadCmpAz()
    staInfos = list()
    with open(fileName,'r') as staFile:
        lines = staFile.readlines()
    strL='1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'
    count=0
    for line in lines:
        infoStr = line.split()
        countN=int(count/len(strL))
        countM=int(count%len(strL))
        nickName=strL[countN]+strL[countM]+strL[countN]+strL[countM]
        dep=0
        if len(infoStr)>=8:
            dep=float(infoStr[7])
        az=0
        azE=0
        netsta=infoStr[0]+'.'+infoStr[1]
        if netsta in cmpAz:
            az,azE=cmpAz[netsta]
        staInfo = {'net':infoStr[0],'sta':infoStr[1],'comp':[infoStr[2]+'E',\
         infoStr[2]+'N', infoStr[2]+'Z'], 'lo': float(infoStr[3]), \
         'la': float(infoStr[4]),'dep':dep, 'nickName':nickName,'az':az,\
         'azE':azE}
        staInfos.append(staInfo)
        count=count+1
    return staInfos

def saveStaInfos(staInfos,fileName):
    with open(fileName,'w+') as f:
        for staInfo in staInfos:
            f.write('%s %s %s %.8f %.8f 0 0 %.8f 0\n'%(staInfo['net'],\
                staInfo['sta'],staInfo['comp'][0][:-1],staInfo['lo'],\
                staInfo['la'],staInfo['dep']))

