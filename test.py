import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import detecQuake
import trainPS
import sacTool
from imp import reload
from obspy import UTCDateTime
import numpy as np
import tool
from glob import glob
from locate import locator
#import pyMFTCuda

taupM=tool.quickTaupModel()
staLstFile='staLst'
staInfos=sacTool.readStaInfos(staLstFile)
laL=[28,35]
loL=[102,106]
laN=20
loN=20
modelL = [trainPS.loadModel('modelPNew2_15'),trainPS.loadModel('modelSNew2_15')]
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= detecQuake.getStaTimeL(staInfos, aMat, taupM=taupM)
quakeLs=list()

for i in range(181, 210):
    print('pick on ',i)
    date = UTCDateTime(2008,1,1)+i*86400
    dayNum=int(date.timestamp/86400)
    dayDir='output/'+str(dayNum)
    if os.path.exists(dayDir):
        print('done')
        continue
    staL = detecQuake.getStaL(staInfos, aMat, staTimeML, modelL, date, taupM=taupM, mode='mid')
    quakeLs.append(detecQuake.associateSta(staL, aMat, staTimeML, timeR=10, maxDTime=3, N=1,\
        locator=locator(staInfos)))
    tool.saveQuakeLs(quakeLs, 'phaseLst2')
    tool.saveQuakeLWaveform(staL, quakeLs[-1], matDir='output/')
    detecQuake.plotResS(staL,quakeLs[-1])

tool.saveQuakeLs(quakeLs, 'phaseLst2')

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import detecQuake
#import trainPS
import sacTool
from imp import reload
from obspy import UTCDateTime
import numpy as np
import tool
from glob import glob
from locate import locator
import pyMFTCuda

taupM=tool.quickTaupModel()
staLstFile='F:\\program\\accuratePicker\\staLst'
staInfos=sacTool.readStaInfos(staLstFile)
laL=[28,35]
loL=[102,106]
laN=20
loN=20
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= detecQuake.getStaTimeL(staInfos, aMat, taupM=taupM)

date = UTCDateTime(2008,1,1)+191*86400
quakeLs=tool.readQuakeLsByP('phaseLst?',staInfos)
quakeL=[]
for quakeLTmp in quakeLs:
    quakeL=quakeL+quakeLTmp
waveformL, tmpNameL=tool.loadWaveformLByQuakeL(quakeLs[2][:],isCut=True)
date = UTCDateTime(2008,1,1)+183*86400
staL = detecQuake.getStaL(staInfos, None, staTimeML, None, date, taupM=taupM, \
    mode='mid',isPre=False)
quakeCCL=pyMFTCuda.doMFTAll(staL,waveformL,date.timestamp,87000*50,\
    locator=locator(staInfos),tmpNameL=tmpNameL,isParallel=True)
