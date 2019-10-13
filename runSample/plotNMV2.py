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
os.environ["MKL_NUM_THREADS"] = "10"
workDir='NM/'
f=[2,8]
R=[37.5,43,95,104.5]
deviceL=['cuda:1']
quakeCCLsName='NM/phaseCCLstNew8'
sDate=UTCDateTime(2014,12,1,0,0,0)
eDate=UTCDateTime(2015,10,1,0,0,0)
#staInfos=sacTool.readStaInfos('staLst_NM_New')
cmpAz=sacTool.loadCmpAz()
staInfos=sacTool.readStaInfos('staLst_NM_New',cmpAz=cmpAz)
loc=locator(staInfos)

quakeLs=tool.readQuakeLs('NM/phaseLstNMALLReloc_removeBadSta',staInfos)
quakeTmpL=tool.readQuakeLs('NM/phaseLstTomoReloc_removeBadSta',staInfos)[0]
quakeCCLs=tool.readQuakeLs('NM/quakeCCLstNewAll',staInfos,isQuakeCC=True)

##plot quakeDis###
figDir=workDir+'fig/'
detecQuake.plotQuakeDis(quakeLs,R=R,staInfos=staInfos,topo='Ordos.grd')
plt.savefig(figDir+'quakeLsDis.png',dpi=500)
#plt.savefig(figDir+'quakeLsDis.eps')
plt.close()

detecQuake.plotQuakeDis([quakeTmpL],R=R,staInfos=staInfos,topo='Ordos.grd',alpha=1,markersize=2)
plt.savefig(figDir+'quakeTmpLDis.png',dpi=500)
#plt.savefig(figDir+'quakeLsDis.eps')
plt.close()

detecQuake.plotQuakeCCDis(quakeCCLs,quakeTmpL,R=R,staInfos=staInfos,topo='Ordos.grd',minCC=0.3)
plt.savefig(figDir+'quakeCCLsDis.png',dpi=500)
#plt.savefig(figDir+'quakeLsDis.eps')
plt.close()