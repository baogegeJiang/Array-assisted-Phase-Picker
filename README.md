# accuratePicker  

---

author: Jiang Yiran & Ning Jieyuan  
a program for earthquakes and micro-earthquakes detection  
it has three fundamental parts：  
1 detect earthquakes using improved phaseNet and array strategy  
2 detect micro-earthquakes based on GPU-WMFT  
3 calculate travel time difference by cross-correlation  and provide input files for tomoDD

As we developed this program in our research, it also contains many other functions.  if not needed, you can ignore the other part. Any way, the reorgainizing work is going on and we would try to give a better way to use for users who need this kind of functions. 

## 1、Improved phaseNet and array strategy  

Needed package in this part: numpy, numba, obspy, h5py, scipy,matplotlib, openpyxl, tensorflow-gpu,(optional: basemap, netCDF4, lxml, pykml,pycpt). And pycpt is not available on open source, then you can install it via:
```
pip install https://github.com/j08lue/pycpt/archive/master.zip
```

phaseNet is based on ZhuW‘s paper and we adopted it for continuous waveform. the array strategy is based on Jiang Y and Ning J's papaer. As we provide some pre-trained model, you can directly use our program on your data.  
in general, you need to provide the following things before runing our program:  
(1) station information list file:  

```
#net sta    comp longitude/° latitude/° elevation/m  rms_of_lon rms_of_lat
 XX  JJS    BH   104.55      31.00      0.000000     0.000000   0 0.000000  
```

 (2) file path function:  
 write a funciton in names.py that you input the net/station/comp/date return the sacFileNames list. we give a example:
```python
def FileName(net, station, comp, YmdHMSJ):
    sacFileNames = list()
    c=comp[-1]
    if c=='Z':
        c='U'
    sacFileNames.append('wlx/data/'+net+'.'+station+'.'+c+'.SAC')
    return sacFileNames
```
net is the network name(e. g. 'XX' ); station is the station name(e. g. 'ABC' ); comp is the component name(e. g. 'BHE' ); YmdHMSJ is a dict contained date information(year, month, day, hour, minute, second, day num from the first day of the year)(e. g. {'Y': '2019', 'm': '01', 'd': '01', 'H': '00', 'M': '01', 'S': '00', 'J': '001'}). as in our example, the function will return the list of file path(e. g. ['wlx/data/XX.ABC.E.SAC'])

(3) edit the run script:
we give the detail in the script
```python
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
laL=[35,45]#area: [min latitude, max latitude]
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
```

then run this script you can detect earthquakes in your data

## Reference:  
Zhu W, Beroza G C. PhaseNet: a deep-neural-network-based seismic arrival-time picking method. Geophys J Int, 2018, 216(1): 261-273.  
Jiang Y R, Ning J Y. Automatic detection of seismic body-wave phases and determination of their arrival times based on support vector machine(in Chinese). Chinese J Geophys, 2019, 62(1): 361-373.  