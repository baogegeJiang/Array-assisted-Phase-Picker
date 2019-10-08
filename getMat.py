import os 
import detecQuake
import sacTool
from imp import reload
from obspy import UTCDateTime
import tool
from glob import glob
import names
R=[37.5,43,95,104.5]
sDate=UTCDateTime(2014,10,17,0,0,0)
eDate=UTCDateTime(2014,12,18,0,0,0)
cmpAz=sacTool.loadCmpAz()
staInfos=sacTool.readStaInfos('staLst_NM',cmpAz=cmpAz)
quakeLs=tool.readQuakeLs('NM/phaseLstNMALLReloc',staInfos)
laL=R[:2]
loL=R[2:]
laN=30
loN=30
modelL = None
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= [None for staInfo in staInfos]
for sec in range(int(sDate.timestamp),int(eDate.timestamp),86400):
    print(sec)
    date=UTCDateTime(sec)
    dayNum=int(sec/86400)
    dayDir='NM/output/'+str(dayNum)
    
    if os.path.exists(dayDir):
        print('done')
        continue
    quakeL=[]
    for quakeLTmp in quakeLs:
        for quake in quakeLTmp:
            if quake.time>=date.timestamp and quake.time<date.timestamp+86400:
                quakeL.append(quake)
    if len(quakeL)==0:
        print('no quake',date)
        continue
    staL = detecQuake.getStaL(staInfos, aMat, staTimeML, modelL,date, \
        getFileName=names.NMFileName,mode='mid',isPre=False,R=R,f=[-1,-1])
    print('quake Num:%d'%len(quakeL))
    tool.saveQuakeLWaveform(staL,quakeL, matDir='NM/output/')
    detecQuake.plotResS(staL,quakeL,outDir='NM/output/')
