from glob import glob
import tool
import sacTool
staInfos=sacTool.readStaInfos('staLstAll')
staFileLst=tool.loadFileLst(staInfos,'fileLst')
compL={'BHE':'3','BHN':'2','BHZ':'1'}
def NMFileName(net, station, comp, YmdHMSJ):
    sacFileNames = list()
    Y = YmdHMSJ
    if net == 'GS' or net=='NM':
            dir = '/media/jiangyr/shanxidata2/nmSacData/'
            staDir = dir+net+'.'+station+'/'
            YmDir = staDir+Y['Y']+Y['m']+'/'
            sacFileNamesStr = YmDir+net+'.'+station+'.'+Y['Y']+Y['m']+Y['d']+'.*'+comp+'*SAC'
            for file in glob(sacFileNamesStr):
                sacFileNames.append(file)
            return sacFileNames
    comp=compL[comp]
    if station in staFileLst:
        if YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH' in staFileLst[station]:
            staDir=staFileLst[station][YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH']
            fileP=staDir+'/??/'+YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'*'+comp+'.m'
            sacFileNames=sacFileNames+glob(fileP)
    return sacFileNames

def NMFileNameHour(net, station, comp, YmdHMSJ):
    sacFileNames = list()
    Y = YmdHMSJ
    if net == 'GS' or net=='NM':
            dir = '/media/jiangyr/shanxidata2/nmSacData/'
            staDir = dir+net+'.'+station+'/'
            YmDir = staDir+Y['Y']+Y['m']+'/'
            sacFileNamesStr = YmDir+net+'.'+station+'.'+Y['Y']+Y['m']+Y['d']+'.*'+comp+'*SAC'
            for file in glob(sacFileNamesStr):
                sacFileNames.append(file)
            return sacFileNames
    comp0=comp
    sacFileNames = list()
    comp=compL[comp]
    if station in staFileLst:
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

def FileName(net, station, comp, YmdHMSJ):
    sacFileNames = list()
    c=comp[-1]
    if c=='Z':
        c='U'
    sacFileNames.append('wlx/data/'+net+'.'+station+'.'+c+'.SAC')
    #print(sacFileNames)
    return sacFileNames

