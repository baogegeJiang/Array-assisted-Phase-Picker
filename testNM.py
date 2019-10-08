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
import names
staLstFile='staLst_NM_New'
workDir='/home/jiangyr/accuratePickerV3/NM/'
R=[35,45,96,105]#[34,42,104,112]
staInfos=sacTool.readStaInfos(staLstFile)
laL=[35,45]
loL=[96,105]
laN=35
loN=35
taupM=tool.quickTaupModel(modelFile='iaspTaupMat')
modelL = [trainPS.loadModel('modelP_320000_100-2-15'),trainPS.loadModel('modelS_320000_100-2-15')]
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= detecQuake.getStaTimeL(staInfos, aMat, taupM=taupM)
quakeLs=list()
bSec=UTCDateTime(2015,6,1).timestamp+0*86400*230
eSec=UTCDateTime(2015,10,1).timestamp+0*86400*231

for date in range(int(bSec),int(eSec),86400):
    dayNum=int(date/86400)
    dayDir=workDir+'output20190901/'+str(dayNum)
    if os.path.exists(dayDir):
        print('done')
        continue
    date=UTCDateTime(float(date))
    print('pick on ',date)
    staL = detecQuake.getStaL(staInfos, aMat, staTimeML, modelL, date, getFileName=names.NMFileName,mode='norm',f=[2,15])
    quakeLs.append(detecQuake.associateSta(staL, aMat, staTimeML, timeR=10, maxDTime=3, N=1,locator=locator(staInfos)))
    tool.saveQuakeLWaveform(staL, quakeLs[-1], matDir=workDir+'output20190901/',\
            index0=-1500,index1=1500)
    tool.saveQuakeLs(quakeLs, workDir+'phaseLsNM20190901V6')
    detecQuake.plotResS(staL,quakeLs[-1],outDir=workDir+'output20190901/')
    staL=[]
