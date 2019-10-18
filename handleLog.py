import re
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
def validMean(vL):
    vM=np.median(vL)
    vD=vL.std()
    vLst=np.where(np.abs(vL-vM)<10*vD)
    vL=vL[vLst]
    return vL

def getLocByLog(filename):
    p1=r"GPS: POSITION.{36}"
    p2=r"Pos=.{23}"
    if not os.path.exists(filename):
        return 999,999,999,999,999,999
    with open(filename) as f:
        try :
            lines=f.read()
        except:
            print('######wrong########')
            return 999,999,999,999,999,999
        else:
            pass
        pRe1=re.compile(p1)
        pRe2=re.compile(p2)
        laL=[]
        loL=[]
        zL=[]
        #print(lines[:100],pRe2.findall(lines))
        for line in pRe1.findall(lines):
            EW=1
            NS=1
            if line[15]=='S':
                NS=-1
            if line[28]=='W':
                EW=-1
            la=NS*(float(line[16:18])+float(line[19:21])/60+float(line[22:27])/3600)
            laL.append(la)
            lo=EW*(float(line[29:32])+float(line[33:35])/60+float(line[36:41])/3600)
            loL.append(lo)
            zL.append(float(line[42:-1]))
        for line in pRe2.findall(lines):
            EW=1
            NS=1
            #3907.25051N11549.00458E
            line=line[4:]
            if line[10]=='S':
                NS=-1
            if line[22]=='W':
                EW=-1
            la=NS*(float(line[:2])+float(line[2:10])/60)
            laL.append(la)
            lo=EW*(float(line[11:14])+float(line[14:-1])/60)
            loL.append(lo)
            zL.append(lo)
    if len(laL)>0 and len(loL)>0:
        laL=np.array(laL)
        loL=np.array(loL)
        zL=np.array(zL)
        laL=validMean(laL)
        loL=validMean(loL)
        zL=validMean(zL)
    if len(laL)>0 and len(loL)>0 and len(zL)>0:
        return laL.mean(), loL.mean(), laL.std(), loL.std(), zL.mean(), zL.std()
    else:
        return 999, 999, 999, 999, 999, 999

def getLocByLogs(filenames):
    laL=[]
    loL=[]
    zL=[]
    for filename in filenames:
        la, lo, laD, loD, z, zD = getLocByLog(filename)
        if laD>1e-3 or loD>1e-3:
            print('RMS too large', laD, loD)
            continue
        if la !=999 and lo!=999:
            laL.append(la)
            loL.append(lo)
            zL.append(z)
    if len(laL)>0 and len(loL)>0:
        laL=np.array(laL)
        loL=np.array(loL)
        zL=np.array(zL)
        return laL.mean(),loL.mean(),laL.std(),loL.std(),zL.mean(),zL.std()
    else:
        return 999,999,999,999,999,999

def getLocByLogsP(p):
    filenames=[];
    for file in glob(p):
        filenames.append(file)
    return getLocByLogs(filenames)

if __name__ == '__main__':
    file='staLoc'
    with open(file,'w+') as f:
        laL=[]
        loL=[]
        dlaL=[]
        dloL=[]
        for staDir in glob('./*/'):
            la,lo,dla,dlo,z,dz=getLocByLogsP(staDir+'/*')
            if dla>10:
                continue
            laL.append(la)
            loL.append(lo)
            dlaL.append(dla)
            dloL.append(dlo)
            f.write('%s %.9f %.9f %.9f %.9f\n'%(os.path.dirname(staDir)[2:],\
                la,lo,dla,dlo))
    plt.plot(np.array(loL),np.array(laL),'.r',markersize=0.5)
    plt.savefig('loc.pdf')
