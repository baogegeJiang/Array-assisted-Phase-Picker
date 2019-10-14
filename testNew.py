import os
import detecQuake
import trainPSV2 as trainPS
import sacTool
from imp import reload
from obspy import UTCDateTime
import tool
from locate import locator
import names

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

workDir='/home/jiangyr/accuratePickerV3/testNew/'# workDir: the dir to save the results
staLstFile='staLst_NM_New'#station list file
bSec=UTCDateTime(2015,6,1).timestamp#begain date
eSec=UTCDateTime(2015,10,1).timestamp# end date
laL=[35,45]#area: [min latitude, max latitude】
loL=[96,105]#area: [min longitude, max longitude]
laN=35 #subareas in latitude
loN=35 #subareas in longitude
nameFunction=names.NMFileName # the function you give in (2)  to get the file path  

#####no need to change########
taupM=tool.quickTaupModel(modelFile='iaspTaupMat')
modelL = [trainPS.loadModel('modelP_320000_100-2-15'),\
trainPS.loadModel('modelS_320000_100-2-15')]
staInfos=sacTool.readStaInfos(staLstFile)
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= detecQuake.getStaTimeL(staInfos, aMat, taupM=taupM)
quakeLs=list()
#############################

for date in range(int(bSec),int(eSec),86400):
    dayNum=int(date/86400)
    dayDir=workDir+'output/'+str(dayNum)
    if os.path.exists(dayDir):
        print('done')
        continue
    date=UTCDateTime(float(date))
    print('pick on ',date)
    staL = detecQuake.getStaL(staInfos, aMat, staTimeML,\
     modelL, date, getFileName=nameFunction,\
     mode='norm',f=[2,15])
    quakeLs.append(detecQuake.associateSta(staL, aMat, \
        staTimeML, timeR=10, maxDTime=3, N=1,locator=\
        locator(staInfos)))
    '''
    save:
    result's in  workDir+'phaseLst'
    result's waveform in  workDir+'output/'
    result's plot picture in  workDir+'output/'
    '''
    tool.saveQuakeLs(quakeLs, workDir+'phaseLst')
    tool.saveQuakeLWaveform(staL, quakeLs[-1], \
        matDir=workDir+'output/',\
            index0=-1500,index1=1500)
    detecQuake.plotResS(staL,quakeLs[-1],outDir=workDir+'output/')
    staL=[]# clear data  to save memory