import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.basemap as basemap
from netCDF4 import Dataset
from scipy import interpolate as interp
from matplotlib import cm
from lxml import etree
from pykml.factory import KML_ElementMaker as KML
from pycpt.load import gmtColormap as cpt2cm
from distaz import DistAz
pi=3.1415927
def genBaseMap(R=[0,90,0,180], topo=None):
    m=basemap.Basemap(llcrnrlat=R[0],urcrnrlat=R[1],llcrnrlon=R[2],\
        urcrnrlon=R[3])
    if topo == None:
        #m.etopo()
        pass
    else:
        plotTopo(m,R,topo=topo)
    return m
def plotOnMap(m, lat,lon,cmd='.b',markersize=0.5,alpha=1,linewidth=0.5,mfc=[]):
    x,y=m(lon,lat)
    if len(mfc)>0:
        return plt.plot(x,y,cmd,markersize=markersize,alpha=alpha,linewidth=linewidth,mfc=mfc)
    else:
        return plt.plot(x,y,cmd,markersize=markersize,alpha=alpha,linewidth=linewidth)

def scatterOnMap(m, lat,lon,s,alpha=1,c=None):
    x,y=m(lon,lat)
    plt.scatter(x,y,s=s,alpha=alpha,c=c)

def readnetcdf(file='Ordos.grd'):
    nc=Dataset(file)
    la=nc.variables['lat'][:]
    lo=nc.variables['lon'][:]
    z=nc.variables['z'][:]
    return np.array(la),np.array(lo),np.array(z)

def getZInR(la0,lo0,z0,R,laN=500,loN=500):
    la=np.arange(R[0],R[1],(R[1]-R[0])/laN)
    lo=np.arange(R[2],R[3],(R[3]-R[2])/loN)
    Z=interp.interp2d(lo0,la0,z0)
    z=Z(lo,la)
    lo,la=np.meshgrid(lo,la)
    return la,lo,z

def plotTopo(m,R,topo='Ordos.grd',laN=800,loN=800,cptfile='wiki-2.0.cpt'):#'cpt17.txt'):
    la0,lo0,z0=readnetcdf(topo)
    la,lo,z=getZInR(la0,lo0,z0,R,laN=laN,loN=loN)
    x,y=m(lo,la)
    #print(la[0,0],lo[0,0])
    m.pcolor(x,y,z,cmap=cpt2cm(cptfile),vmin=0, vmax=5000)
    plt.colorbar()
    #z.set_clim(-9000,9000)

def quakeLs2kml(quakeLs,filename):
    fold=KML.Folder()
    for quakeL in quakeLs:
        for quake in quakeL:
            fold.append(lalodep2Place(quake.loc[0],quake.loc[1],quake.loc[2]))
    content = etree.tostring(etree.ElementTree(fold),pretty_print=True)
    #print(type(content))
    with open(filename,'wb') as fp:
        fp.write(content)

def lalodep2Place(la,lo,dep):
    return KML.Placemark(KML.Point(\
        KML.coordinates(str(lo)+','+str(la)+','+str(dep))))

class Fault:
    def __init__(self,R=None,laL=[],loL=[],strike=None,dip=None,angle=None):
        self.R=R
        self.laL=laL
        self.loL=loL
        self.dip=dip
        self.angle=angle
        self.strike=strike
    def update(self):
        laL=np.array(self.laL)
        loL=np.array(self.loL)
        self.R=[laL.min(),laL.max(),loL.min(),loL.max()]
    def inR(self,R0):
        R=self.R
        if (R[1]<R0[0] or R[0]>R0[1]) or (R[3]<R0[2] or R[2]>R0[3]) :
            return False
        else:
            return True
    def plot(self,m=None,cmd='-k',markersize=0.5,alpha=1,isDip=False,l=0.3,linewidth=0.5):
        laL=np.array(self.laL)
        loL=np.array(self.loL)
        dipLaL=[]
        dipLoL=[]
        cmd0='r'
        if self.strike!=None:
            la0=laL[int(len(laL)/2)]
            lo0=loL[int(len(loL)/2)]
            if self.angle!=None:
                l=l*np.sin(self.angle)
            dipLaL=np.array([0,l*np.cos(self.strike+pi/2)])+la0
            dipLoL=np.array([0,l*np.sin(self.strike+pi/2)])+lo0
        if m!=None:
            return plotOnMap(m,laL,loL,cmd,markersize,alpha,linewidth=linewidth)
            if isDip and len(dipLaL)>0:
                plotOnMap(m,dipLaL,dipLoL,cmd0,markersize,alpha,linewidth=linewidth)
        else:
            return plt.plot(loL,laL,cmd,markersize=markersize,alpha=alpha)
            if isDip and len(dipLaL)>0:
                plt.plot(dipLoL,dipLaL,cmd0,markersize=markersize,alpha=alpha)

def readFault(filename):
    faultL=[]
    strikeD={'N':0,'NNE':pi/8*1,'NE':pi/8*2,'NEE':pi/8*3,'E':pi/8*4\
    ,'SEE':pi/8*5,'SE':pi/8*6,'SSE':pi/8*7,'S':pi/8*8,'SSW':pi/8*9,\
    'SW':pi/8*10,'SWW':pi/8*11,'W':pi/8*12,'NWW':pi/8*13,'NW':pi/8*14\
    ,'NNW':pi/8*15}
    with open(filename) as f:
        for line in f.readlines():
            line=line.split()
            if line[0]=='>':
                faultL.append(Fault(laL=[],loL=[]))
            if line[0]=='#':
                line1=line[1].split('|')
                if len(line1)>=25:
                    for i in range(5,8):
                        strikeStr=line1[i]
                        if len(strikeStr)>0:
                            try:
                                if strikeStr in strikeD:
                                    faultL[-1].dip=strikeD[strikeStr]
                                else:
                                    strikeL=strikeStr.split('-')
                                    tmp=0
                                    ct=0
                                    for strike in strikeL:
                                        if len(strike)!=0:
                                            ct+=1
                                            tmp+=tmp+float(strike)
                                    if ct!=0:
                                        if i==5:
                                            faultL[-1].strike=tmp/ct/180*pi
                                        if i==6:
                                            faultL[-1].dip=tmp/ct/180*pi
                                        if i==7:
                                            faultL[-1].angle=tmp/ct/180*pi
                            except:
                                pass
                            else:
                                pass
            if line[0]!='>' and line[0]!='#':
                faultL[-1].laL.append(float(line[1]))
                faultL[-1].loL.append(float(line[0]))
    for fault in faultL:
        fault.update()
    return faultL

class lineArea:
    def __init__(self,p0,p1,dkm):
        self.p0=np.array(p0)
        self.p1=np.array(p1)
        self.lakm=DistAz(p0[0],p0[1],p0[0]+1,p0[1]).getDelta()*111.19
        self.lokm=DistAz(p0[0],p0[1],p0[0],p0[1]+1).getDelta()*111.19
        self.dkm=dkm
        xy1=self.xy(p1)
        self.len=np.linalg.norm(xy1)
        self.v=(xy1)/self.len
    def xy(self,p):
        xy=np.zeros(2)
        xy[0]=(p[0]-self.p0[0])*self.lakm
        xy[1]=(p[1]-self.p0[1])*self.lokm
        return xy
    def convert(self, p, isIn=True):
        h=-999
        dp=self.xy(p)
        h=(dp*self.v).sum()
        if (h<0 or h>self.len) and isIn:
            return -999
        dkm=np.abs(self.calDkm(p))
        if dkm>self.dkm:
            return -999
        return h
    def calDkm(self,p):
        dp=self.xy(p)
        return -(dp[0]*self.v[1]-dp[1]*self.v[0])

def plotInline(quakeLs,p0,p1,dkm=50,mul=3,along=True,alpha=0.3,minSta=10,staInfos=None,minCover=0.7):
    line=lineArea(p0,p1,dkm)
    hL=[]
    depL=[]
    mlL=[]
    dkmL=[]
    timeL=[]
    for quakeL in quakeLs:
        for quake in quakeL:
            if len(quake)<minSta:
                continue
            loc=quake.loc
            h=line.convert(loc[:2])
            if h>0:
                if staInfos != None:
                    if quake.calCover(staInfos)<minCover:
                        continue
                hL.append(h)
                depL.append(loc[2])
                mlL.append(max(0.1,quake.ml))
                dkmL.append(line.calDkm(loc[:2]))
                timeL.append(quake.time%86400)
                if quake.loc[2]>70 and quake.ml>2:
                    print(quake.time,quake.loc)
    hL=np.array(hL)
    depL=np.array(depL)
    mlL=np.array(mlL)
    dkmL=np.array(dkmL)
    timeL=np.array(timeL)
    if along:
        plt.scatter(hL,depL,mlL*mul,c=timeL,alpha=alpha)
    else:
        plt.scatter(dkmL,depL,mlL*mul,c=timeL,alpha=alpha)
    ax = plt.gca()
    ax.invert_yaxis()



def plotQuakeDis(quakeLs,output='quakeDis.png',cmd='.b',markersize=0.8,\
    alpha=0.3,R=None,topo=None,m=None,staInfos=None,minSta=8,minCover=0.8,\
    faultFile="Chinafault_fromcjw.dat",mul=1,loL0=[],laL0=[],isBox=False):
    la=[]
    lo=[]
    dep=[]
    mlL=[]
    plt.close()
    plt.figure(figsize=[12,8])
    for quakeL in quakeLs:
        for quake in quakeL:
            if len(quake)<minSta:
                continue
            if staInfos!=None:
                if quake.calCover(staInfos)<minCover:
                    continue
            ml=0
            if quake.ml !=None:
                if quake.ml>-2:
                    ml=quake.ml
            la.append(quake.loc[0])
            lo.append(quake.loc[1])
            dep.append(quake.loc[2])
            mlL.append(ml)
    la=np.array(la)
    lo=np.array(lo)
    dep=np.array(dep)
    mlL=np.array(mlL)
    if R==None:
        R=[la.min(),la.max(),lo.min(),lo.max()]
    #print(R)
    
    if m==None:
        m=genBaseMap(R=R,topo=topo)
    if not staInfos == None:
        sla=[]
        slo=[]
        sdep=[]
        for staInfo in staInfos:
            sla.append(staInfo['la'])
            slo.append(staInfo['lo'])
            sdep.append(staInfo['dep'])
        sla=np.array(sla)
        slo=np.array(slo)
        sdep=np.array(sdep)
        hS,=plotOnMap(m,sla,slo,'^r',markersize=5,alpha=1,linewidth=1)
    faultL=readFault(faultFile)
    hF=None
    for fault in faultL:
        if fault.inR(R):
            hFTmp,=fault.plot(m)
            if hFTmp!=None:
                hF=hFTmp
    if len(laL0)>1:
        hC,=plotOnMap(m,laL0,loL0,'ok',markersize=2,mfc=[1,1,1])
    if isBox:
        laLB0=[38.7,42.2]
        loLB0=[97.5,103.8]
        laLB=np.array([laLB0[0],laLB0[0],laLB0[1],laLB0[1],laLB0[0]])
        loLB=np.array([loLB0[1],loLB0[0],loLB0[0],loLB0[1],loLB0[1]])
        plotOnMap(m,laLB,loLB,'r',linewidth=3,markersize=3)
    hQ,=plotOnMap(m,la,lo,cmd,markersize,alpha)
    #mt.scatterOnMap(m,la,lo,s=np.exp(mlL/1.5)*mul,alpha=alpha,c=np.array([1,0,0]))
    parallels = np.arange(0.,90,2.)

    if len(laL0)>1:
        #hC,=mt.plotOnMap(m,laL0,loL0,'ok',markersize=2)
        plt.legend((hQ,hC,hS,hF),('Quakes','Catalog','Station','Faults'),\
            bbox_to_anchor=(1, 1),loc='lower right')
    else:
        plt.legend((hQ,hS,hF),('Quakes','Station','Faults'),bbox_to_anchor=(1, 1),\
              loc='lower right')
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(10.,360.,2.)
    plt.gca().yaxis.set_ticks_position('left')
    m.drawmeridians(meridians,labels=[True,False,False,True])
    plt.savefig(output)
    return m

def plotQuakeCCDis(quakeCCLs,quakeRefL,output='quakeDis.png',cmd='.r',markersize=0.8,\
    alpha=0.3,R=None,topo=None,m=None,staInfos=None,minSta=8,minCover=0.8,\
    faultFile="Chinafault_fromcjw.dat",mul=1,minCC=0.5):
    la=[]
    lo=[]
    dep=[]
    mlL=[]
    count=0
    for quakeL in quakeCCLs:
        for quake in quakeL:
            if len(quake)<minSta:
                continue
            if staInfos!=None:
                if quake.calCover(staInfos,minCC=minCC)<minCover:
                    continue
            ml=0
            if quake.ml !=None:
                if quake.ml>-2:
                    ml=quake.ml
            la.append(quake.loc[0])
            lo.append(quake.loc[1])
            dep.append(quake.loc[2])
            mlL.append(ml)
            count+=1
    print(count)
    la=np.array(la)
    lo=np.array(lo)
    dep=np.array(dep)
    mlL=np.array(mlL)
    if R==None:
        R=[la.min(),la.max(),lo.min(),lo.max()]
    laR=[]
    loR=[]
    depR=[]
    mlLR=[]
    for quake in quakeRefL:
        ml=0
        if quake.ml !=None:
            if quake.ml>-2:
                ml=quake.ml
        laR.append(quake.loc[0])
        loR.append(quake.loc[1])
        depR.append(quake.loc[2])
        mlLR.append(ml)
    laR=np.array(laR)
    loR=np.array(loR)
    depR=np.array(depR)
    mlLR=np.array(mlLR)
    if m==None:
        m=genBaseMap(R=R,topo=topo)
    if not staInfos == None:
        sla=[]
        slo=[]
        sdep=[]
        for staInfo in staInfos:
            sla.append(staInfo['la'])
            slo.append(staInfo['lo'])
            sdep.append(staInfo['dep'])
        sla=np.array(sla)
        slo=np.array(slo)
        sdep=np.array(sdep)
        hS,=plotOnMap(m,sla,slo,'^k',markersize=5,alpha=1)
    faultL=readFault(faultFile)
    hF=None
    for fault in faultL:
        if fault.inR(R):
            hFTmp,=fault.plot(m)
            if hFTmp!=None:
                hF=hFTmp

    hT,=plotOnMap(m,laR,loR,'*b',markersize*2,1)
    hCC,=plotOnMap(m,la,lo,cmd,markersize,alpha)
    print(len(laR),len(la))

    plt.legend((hT,hCC,hS,hF),('Templates','Microearthquakes','Station','Faults'))
    #mt.plotOnMap(m,laR,loR,'*k',markersize*2,1)
    #mt.scatterOnMap(m,la,lo,s=np.exp(mlL/1.5)*mul,alpha=alpha,c=np.array([1,0,0]))
    plt.title('minSta:%d minCover:%.1f minCC:%.1f MFT:%d'%(minSta,minCover,minCC,count))
    dD=max(int((R[1]-R[0])*10)/40,int((R[3]-R[2])*10)/40)
    parallels = np.arange(int(R[0]),int(R[1]+1),dD)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(int(R[2]),int(R[3]+1),dD)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    return m