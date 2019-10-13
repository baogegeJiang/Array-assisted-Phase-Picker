import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import detecQuake
import sacTool
from imp import reload
from obspy import UTCDateTime
import numpy as np
import tool
from glob import glob
from locate import locator
import matplotlib.pyplot as plt
import mapTool as mt
import tomoDD
import locate
import iterconv
import locate
import matplotlib.pyplot as plt
import pyMFTCuda
import names
import time
import torch

staInfos=sacTool.readStaInfos('staLst_NM')

R=[37.5,43,95,104.5]
RML=[38.7,39.3,101.8,102]
loc=locate.locator(staInfos)
quakeTomoRelocLs=tool.readQuakeLs('NM/phaseLstTomoReloc',staInfos)
quakeWLXL=tool.selectQuake(quakeTomoRelocLs,RML,staInfos,minSta=8,maxCount=5,outDir='NM/output/',\
    minCover=0.7,laN=5,loN=5)
quakeWLXL=tool.selectQuake([quakeTomoL],RML,staInfos,minSta=6,maxCount=5,outDir='NM/output/',\
    minCover=0.7,laN=5,loN=5)
for quake in quakeWLXL:
    loc.locate(quake,isDel=True)

tool.saveQuakeLs([quakeWLXL],'NM/phaseLstWLX')
WLXDIR='/home/jiangyr/MatchLocate/NM2/'
tmpDir=WLXDIR+'Template/'

for quake in quakeWLXL:
    sacTool.cutSacByQuakeForCmpAz(quake,staInfos,names.NMFileNameHour,R=[38,40,100,104],outDir=tmpDir,\
        decMul=2,B=0,E=70,isFromO=True,nameMode='ML',maxDT=30)
traceDir=WLXDIR+'Trace/'
date=UTCDateTime(2015,9,19,0,0,10)
sacTool.cutSacByDate(date,staInfos,names.NMFileName,R=[38,40,100,104],outDir=traceDir,\
    decMul=2,B=0,E=86400-10,isFromO=True,nameMode='ML')

traceDir=WLXDIR+'Trace/'
date=UTCDateTime(2015,1,1,0,0,10)
for i in range(30):
    date+=86400
    sacTool.cutSacByDate(date,staInfos,names.NMFileName,R=[38,40,100,104],outDir=traceDir,\
    decMul=2,B=0,E=86400-10,isFromO=True,nameMode='ML')

quakeWLXL=tool.readQuakeLsByP('NM/phaseLstWLX',staInfos)[0]
catalog=WLXDIR+'catalog.dat'
tool.saveQuakeLs(quakeWLXL,catalog,mod='ML')
date=UTCDateTime(2015,9,19,0,0,10)
laL=R[:2]
loL=R[2:]
laN=30
loN=30
modelL = None
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= [None for staInfo in staInfos]
staL = detecQuake.getStaL(staInfos, aMat, staTimeML, modelL,date, \
    getFileName=names.NMFileName,mode='mid',isPre=False,f=[2,8],R=[38,40,100,104])
waveformL,tmpNameL=tool.loadWaveformLByQuakeL(quakeWLXL,isCut=True,matDir='NM/output/',f=[2,8])
quakeCCLs=[]
quakeCCLs.append(pyMFTCuda.doMFTAll(staL,waveformL,date.timestamp,86400*50,\
    locator=locator(staInfos),tmpNameL=tmpNameL,minDelta=50*5,MINMUL=7,\
    quakeRefL=quakeWLXL,maxCC=0.3,winTime=0.8,R=[38,40,100,104],minMul=3,maxDis=200,\
    deviceL=['cuda:0','cuda:1']))
date=UTCDateTime(2015,1,1,0,0,10)
quakeCCLs=[]
for i in range(30):
    date+=86400
    staL = detecQuake.getStaL(staInfos, aMat, staTimeML, modelL,date, \
        getFileName=names.NMFileName,mode='mid',isPre=False,f=[2,8],R=[38,40,100,104])
    quakeCCLs.append(pyMFTCuda.doMFTAll(staL,waveformL,date.timestamp,86400*50,\
    locator=locator(staInfos),tmpNameL=tmpNameL,minDelta=50*5,MINMUL=7,\
    quakeRefL=quakeWLXL,maxCC=0.3,winTime=0.4,R=[38,40,100,104],minMul=3,maxDis=200))
    tool.saveQuakeLWaveform(staL, quakeCCLs[-1], matDir='NM/outputCC/')
    tool.saveQuakeLs(quakeCCLs, 'NM/phaseCCLst')
    detecQuake.plotResS(staL,quakeCCLs[-1],outDir='NM/outputCC/') 
quakeCCLs=tool.readQuakeLsByP('NM/phaseCCLst',staInfos,isQuakeCC=True)
quakeLD=tool.getQuakeLD(quakeWLXL)
detecQuake.plotQuakeCCDis(quakeCCLs,quakeWLXL,R=R,staInfos=staInfos,alpha=1,mul=0.1,markersize=0.5)
plt.savefig('NM/quakeCCLDis.png',dpi=400)
timeG=loc.getGRef(quakeCCLs[0][0],quakeLD[quakeCCLs[0][0].tmpName])
G,V,v=loc.getGMRef(quakeCCLs[0][0],quakeLD[quakeCCLs[0][0].tmpName])

tool.saveQuakeLWaveform(staL, quakeCCLs[-1], matDir='wlx/outputCC/')
tool.saveQuakeLs(quakeCCLs, 'wlx/phaseCCLst')
detecQuake.plotResS(staL,quakeCCLs[-1],outDir='wlx/outputCC/')
quakeCCLWLX=tool.readQuakeLs(WLXDIR+'MultipleTemplate/DetectedFinal.dat',\
    staInfos,mode='byWLX',tmpNameL=tmpNameL,isQuakeCC=True)
tool.saveQuakeLs([quakeCCLWLX],'NM/phaseLstCCLWLX')
tool.saveQuakeLs(quakeCCLs,'NM/phaseLstCCLNM')
#R=[37.5,43,95,104.5]
#RML=[38.7,39.3,101.8,102]
R=[38,40,100,104]
staInfos=sacTool.readStaInfos('wlx/staLst_wlx')

loc=locator(staInfos)
quakeWLXL=tool.readQuakeLs('wlx/phaseLst',staInfos)[0]

WLXDIR='/home/jiangyr/MatchLocate/NM/'
tmpDir=WLXDIR+'Template/'
for quake in quakeWLXL:
    sacTool.cutSacByQuakeForCmpAz(quake,staInfos,names.FileName,outDir=tmpDir,\
        decMul=2,B=0,E=70,isFromO=True,nameMode='ML')
catalog=WLXDIR+'catalog.dat'
tool.saveQuakeLs(quakeWLXL,catalog,mod='ML')
traceDir=WLXDIR+'Trace/'
date=UTCDateTime(2012,9,2,0,0,0)
sacTool.cutSacByDate(date,staInfos,names.FileName,outDir=traceDir,\
        decMul=2,B=3*3600+1,E=4*3600-1,nameMode='ML')

