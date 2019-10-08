import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import detecQuake
import trainPS
import sacTool
from imp import reload
from obspy import UTCDateTime
import numpy as np
import tool
from glob import glob
from locate import locator
staLstFile='staLstAll'
R=[34,42,104,116]#[34,42,104,112]
staInfos=sacTool.readStaInfos(staLstFile)
tool.getStaInArea(staInfos,'staLst_test',R)
staInfos=sacTool.readStaInfos('staLst_test')
staFileLst=tool.loadFileLst(staInfos,'fileLst')
compL={'BHE':'3','BHN':'2','BHZ':'1'}
def NMFileName(net, station, comp, YmdHMSJ):
    #dir='tmpSacFile/'
    comp0=comp
    sacFileNames = list()
    comp=compL[comp]
    if YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH' in staFileLst[station]:
        staDir=staFileLst[station][YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH']
        fileP=staDir+'/??/'+YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'*'+comp+'.m'
        sacFileNames=sacFileNames+glob(fileP)
    if len(sacFileNames)==0:
        sacDir='/media/jiangyr/Hima_Bak/hima31/'
        fileP=sacDir+YmdHMSJ['Y']+YmdHMSJ['m']+YmdHMSJ['d']+\
        '.'+YmdHMSJ['J']+'*/*.'+station+'*.'+comp0
        sacFileNames=sacFileNames+glob(fileP)
    if len(sacFileNames)==0:
        sacDir='/media/jiangyr/shanxidata2/hima31_2/'
        fileP=sacDir+YmdHMSJ['Y']+YmdHMSJ['m']+YmdHMSJ['d']+\
        '.'+YmdHMSJ['J']+'*/*.'+station+'*.'+comp0
        sacFileNames=sacFileNames+glob(fileP)
    return sacFileNames

taupM=tool.quickTaupModel(modelFile='iaspTaupMat')
laL=[32,44]
loL=[102,118]
laN=40
loN=40
modelL = [trainPS.loadModel('modelPNew2_15'),trainPS.loadModel('modelSNew2_15')]
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= detecQuake.getStaTimeL(staInfos, aMat, taupM=taupM)
quakeLs=list()
bSec=UTCDateTime(2014,1,1).timestamp+0*86400*230
eSec=UTCDateTime(2017,2,4).timestamp+0*86400*231

for date in range(int(bSec),int(eSec),86400):
    dayNum=int(date/86400)
    dayDir='output/'+str(dayNum)
    if os.path.exists(dayDir):
        print('done')
        continue
    date=UTCDateTime(float(date))
    print('pick on ',date)
    #date = UTCDateTime(2015,1,1)+i*86400
    staL = detecQuake.getStaL(staInfos, aMat, staTimeML, modelL, date, getFileName=NMFileName,mode='mid')
    quakeLs.append(detecQuake.associateSta(staL, aMat, staTimeML, timeR=10, maxDTime=3, N=1,locator=locator(staInfos)))
    tool.saveQuakeLWaveform(staL, quakeLs[-1], matDir='output/')
    tool.saveQuakeLs(quakeLs, 'phaseLstHimaNewV100')
    detecQuake.plotResS(staL,quakeLs[-1])
    staL=[]
