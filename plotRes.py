import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
R=[34,42,104,116]
cmpAz=sacTool.loadCmpAz()
staInfos=sacTool.readStaInfos('staLst_test',cmpAz=cmpAz)

staInfos=sacTool.readStaInfos('staLst_NM_New',cmpAz=cmpAz)

taupM=tool.quickTaupModel(modelFile='iaspTaupMat.mat')
quakeCCLs=list()
quakeLs=tool.readQuakeLsByP('phaseLstDir/phaseLstAll*',staInfos)
quakeCCLs=tool.readQuakeLsByP('NM/phaseCCLstNew?',staInfos)
quakeL=[]
R=[35,45,96,105]
for quakeLTmp in quakeLs:
    quakeL=quakeL+quakeLTmp
detecQuake.plotQuakeCCDis(quakeCCLs,quakeL,R=R,staInfos=staInfos,alpha=1,mul=0.1,markersize=0.05)

mt.plotInline([quakeL],[39,106],[40,107])
mt.plotInline([quakeL],[39,106],[40,107],along=False)
loc=locator(staInfos)
for quake in quakeL:
    quake,res=loc.locate(quake)
    print(res)
quakeTomoL=tool.selectQuake([quakeL],R,staInfos,minSta=12,maxCount=30)
tool.saveQuakeLs([quakeTomoL],'phaseLstDir/phaseLstTomo')
quakeL=tool.readQuakeLsByP('phaseLstDir/phaseLstTomo',staInfos)[0]
waveformTomoL,tmpNameL=tool.loadWaveformLByQuakeL(quakeTomoL,isCut=True)
dTM=tomoDD.calDTM(quakeTomoL,waveformTomoL,staInfos,maxD=0.3,minC=0.7,minSameSta=5)
dTM=tomoDD.loadDTM()
tomoDD.plotDT(waveformTomoL,dTM,0,132,staInfos)
tomoDD.preEvent(quakeL,staInfos)
tomoDD.preABS(quakeL,staInfos)
tomoDD.preMod(R)
tomoDD.preSta(staInfos)
tomoDD.preDTCC(quakeL,staInfos,dTM)
quakeRelocL=tomoDD.getReloc(quakeL)
tool.saveQuakeLs([quakeRelocL],'phaseLstDir/phaseLstTomoReloc')
detecQuake.plotQuakeDis([quakeRelocL],R=R,staInfos=staInfos,topo='Ordos.grd')
detecQuake.plotQuakeDis([quakeRelocL],R=R,staInfos=staInfos,topo='Ordos.grd',alpha=0.8)
quakeTomoL=tomoDD.selectQuake([quakeRelocL],R,staInfos,minSta=12,maxCount=30)
tool.saveQuakeLs([quakeTomoL],'phaseLstDir/phaseLstTomoV2')
waveformTomoL,tmpNameL=tool.loadWaveformLByQuakeL(quakeTomoL,isCut=True)
dTM=tomoDD.calDTM(quakeTomoL,waveformTomoL,staInfos,maxD=0.3,minC=0.7,minSameSta=5)
tomoDD.saveDTM(dTM,'dTMV2')
#staLstFile='staLst'
#R=[34,42,104,112]
#staInfos=sacTool.readStaInfos(staLstFile)
#tool.getStaInArea(staInfos,'staLst000000',R)
#staInfos=sacTool.readStaInfos('staLst_test')
mt.plotInline([quakeRelocL],[40.5881,110],[40,95,111.9],along=False)
quakeRefs=tool.readQuakeLsByP('phaseLstDir/phaseLstTomoRelocV2',staInfos)[0]
quakeLs=tool.readQuakeLsByP('phaseLstDir/phaseLstAll',staInfos)
quakeRelocL=locate.relocQuakeLs(quakeLs,quakeRefs,staInfos)
#quakeTomoL=tomoDD.selectQuake([quakeL],R,staInfos,minSta=12,maxCount=30)
waveformTomoL,tmpNameL=tool.loadWaveformLByQuakeL(quakeL)
m=tool.calConvM(waveformTomoL[0]['pWaveform'][0,:,:])
np.linalg.eig(m)
quakeLs=tool.readQuakeLsByP('phaseLstDir/phaseLstTomoRelocAllV2',staInfos)
quakeRfL=tool.readQuakeLsByP('phaseLstDir/selectGlobWithTime',staInfos)[0][:100]
quakeRfL=tool.selectQuake(quakeLs,[39.5,39.7,106.8,107],staInfos,minSta=8,maxCount=30,maxDep=5)
waveformRfL,tmpNameL=tool.loadWaveformLByQuakeL(quakeRfL[:],matDir='outputRF/',\
    isCut=True,index0=-500,index1=50*25)
waveformRfL=tool.resampleWaveformL(waveformRfL,5)
staRF,staNum=iterconv.rfOnQuakeL(quakeRfL,waveformRfL,staInfos,mod='iterconv',indexL=np.arange(-0,15*10),isPlot='True')
iterconv.rfOnQuake(quakeRfL[0],waveformRfL[0],staInfos,mod='iterconv',\
    indexL=np.arange(0,15*10),isPlot='True',isFFTShift=False)
for i in range(20):
    tool.plotSpec(waveformRfL[i],plotS=True)


mt.plotInline(quakeLs,[40.72,108.29],[40.96,112.053],along=False)
quakeLs=tool.readQuakeLs()
quakeLsNew=tool.selectQuakeByDis(quakeLs,R,staInfos,minDis=30,maxDis=80,\
    bTime=UTCDateTime(2014,1,1).timestamp,eTime=UTCDateTime(2017,1,1).timestamp,minMl=6)

staFileLst=tool.loadFileLst(staInfos,'fileLst')
compL={'BHE':'3','BHN':'2','BHZ':'1'}
def NMFileName(net, station, comp, YmdHMSJ):
    #dir='tmpSacFile/'
    comp0=comp
    sacFileNames = list()
    comp=compL[comp]
    if YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH' in staFileLst[station]:
        staDir=staFileLst[station][YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH']
        fileP=staDir+'/'+YmdHMSJ['H']+'/'+YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'*'+comp+'.m'
        sacFileNames=sacFileNames+glob(fileP)
        Hour=(int(YmdHMSJ['H'])+1)%24
        H='%02d'%Hour
        fileP=staDir+'/'+H+'/'+YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'*'+comp+'.m'
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

tool.analysis(quakeLs,staInfos)

iterconv.showStaRf(staRF,staNum,staInfos,R)


R=[37,39,108,112]
cmpAz=sacTool.loadCmpAz()
staInfos=sacTool.readStaInfos('staLstAll',cmpAz=cmpAz)
quakeLs=tool.readQuakeLs('aa.ndk',staInfos,mode='NDK')
#quakeLs=tool.readQuakeLs('irisLog',staInfos,mode='IRIS')
quakeL=tool.selectQuakeByDis(quakeLs,R,staInfos,minDis=30,\
    maxDis=90,bTime=UTCDateTime(2014,1,1).timestamp,\
    eTime=UTCDateTime(2017,2,4).timestamp,minMl=6)
tool.saveQuakeLs([quakeL],'phaseLstDir/cmpAzCatalog')
for quake in quakeL:
    sacTool.cutSacByQuake(quake,staInfos,NMFileName,R=R)

sacTool.preCmpAzLog(quakeL)
for quake in quakeL:
    sacTool.cutSacByQuakeForCmpAz(quake,staInfos[:],NMFileName)
sacTool.runCmpAz(staInfos)
sacTool.readCmpAzResult(staInfos)
#a['day'][0,5][0][0,2][2]
