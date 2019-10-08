import argparse
import matplotlib.pyplot as plt
import obspy
import sys
import math
import scipy.io as sio
import scipy
import numpy as np
from numpy import cos, sin
import os
from genMV3 import genModel,genModelSoft
from keras import backend as K
from keras.models import Model
import h5py
import tensorflow as tf
import logging
from sacTool import getDataByFileName
from glob import glob
import obspy
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session =tf.Session(config=config)
K.tensorflow_backend.set_session(session) 

def loadModel(modelFile):
    model = genModel()
    model.load_weights(modelFile)
    return model
class modelPhase:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(processX(X))


def predict(model, X):
    return model.predict(processX(X))


def predictLongData(model, x, N=2000, indexL=range(750, 1250)):
    if len(x) == 0:
        return np.zeros(0)
    N = x.shape[0]
    Y = np.zeros(N)
    perN = len(indexL)
    loopN = int(math.ceil(N/perN))
    perLoop = int(1000)
    inMat = np.zeros((perLoop, 2000, 1, 3))
    for loop0 in range(0, int(loopN), int(perLoop)):
        loop1 = min(loop0+perLoop, loopN)
        for loop in range(loop0, loop1):
            i = loop*perN
            sIndex = min(max(0, i), N-2000)
            if sIndex > 0:
                inMat[loop-loop0, :, :, :] = processX(x[sIndex: sIndex+2000, :])\
                .reshape([2000, 1, 3])
        outMat = model.predict(inMat).reshape([-1, 2000])
        for loop in range(loop0, loop1):
            i = loop*perN
            sIndex = min(max(0, i), N-2000)
            if sIndex > 0:
                Y[indexL[0]+sIndex: indexL[-1]+1+sIndex] = \
                outMat[loop-loop0, indexL].reshape([-1])
    return Y


def processX(X, rmean=True, normlize=True, reshape=True,isNoise=False):
    if reshape:
        X = X.reshape(-1, 2000, 1, 3)
    if rmean:
        X-= X.mean(axis=1,keepdims=True)
    if normlize:
        X /=(X.std(axis=(1, 2, 3),keepdims=True))
    if isNoise:
        X+=(np.random.rand(X.shape[0],2000,1,3)-0.5)*np.random.rand(X.shape[0],1,1,3)*X.max(axis=(1,2,3),keepdims=True)*0.15*(np.random.rand(X.shape[0],1,1,1)<0.1)
    return X


def processY(Y):
    return Y.reshape(-1, 2000, 1, 1)


def trainP(N, SCN):
    matFileTest = 'SC.mat'
    matFileX = 'PX.mat'
    matFileY = 'PY.mat'
    resFile ='resDataP_%d_%d-2-15-with.mat'%(N,SCN)
    N1 = 200
    N2 = 00
    modelFile = 'modelP_%d_%d-2-15-with'%(N,SCN)
    train(matFileX, matFileY, matFileTest, modelFile, resFile, 'px', 'py', N, \
        N1, N2, sN0=100, sN1=300, SCN=SCN)


def trainS(N, SCN):
    matFileTest = 'SC.mat'
    matFileX = 'SX.mat'
    matFileY = 'SY.mat'
    resFile = 'resDataS_%d_%d-2-15-with.mat'%(N,SCN)
    N1 = 300
    N2 = 50
    modelFile = 'modelS_%d_%d-2-15-with'%(N,SCN)
    train(matFileX, matFileY, matFileTest, modelFile, resFile, 'sx', 'sy', N, \
        N1, N2, sN0=200, sN1=200, SCN=SCN)


def shuffleData(X,N=1000):
    n=np.size(X,0)
    L=np.arange(n)
    np.random.shuffle(L[N:-1])
    return L

def getBadDataL(sacLstFile):
    with open(sacLstFile) as f:
        lines=f.readlines()
    badSacDataL=[]
    badLL=[]
    for line in lines:
        sacFileNameL=[[fileName] for fileName in line.split()]
        badSac=getDataByFileName(sacFileNameL,freq=[2,15])
        badSacDataL.append(badSac.data)
        badL=[]
        sacL=[obspy.read(fileName)[0]for fileName in line.split() ]
        for sact in sacL:
            for i in range(10):
                if not 't'+str(int(i)) in sact.stats.sac:
                    continue
                time=sact.stats.sac['t'+str(int(i))]+\
                        sact.stats.starttime.timestamp-badSac.bTime.timestamp
                badL.append(int(time/0.02))
        badLL.append(badL)
    return badSacDataL,badLL



def train(matFileX, matFileY, matFileTest, modelFile, resFile, \
        xStr, yStr, N, N1, N2, sN0=200, sN1=200, SCN=100):
    badSacDataL,badLL=getBadDataL('badSac/fileLst')
    logger=logging.getLogger(__name__)
    N0=N+13010
    model = genModel(xStr[0])
    model0 = model.get_weights()
    print(model.summary())
    eN0=sN0+2000
    eN1=sN1+2000
    matLoad = sio.loadmat(matFileTest)
    dataXTest = matLoad[xStr]
    if yStr=='sy':
        dataYTest=matLoad['sY']
    else:
        dataYTest=matLoad[yStr]

    matLoadY = h5py.File(matFileY)
    dataY = matLoadY[yStr][:N0]

    matLoadX = h5py.File(matFileX)
    dataX = matLoadX[xStr][:N0]
    dataXTest=processX(dataXTest,reshape=False,normlize=False)
    dataX=processX(dataX,reshape=False,normlize=False)
    sIndexTest = 1000
    eIndexTest = 11000
    L = shuffleData(dataXTest)
    dataXTest = dataXTest[L].reshape(-1,3000,1,3)
    dataYTest = dataYTest[L].reshape(-1,3000,1,1)
    L = shuffleData(dataX, N=eIndexTest)
    dataX = dataX[L].reshape(-1,3000,1,3)#, :, :, :]
    dataY = dataY[L].reshape(-1,3000,1,1)#, :]
    SCN0=dataXTest.shape[0]
    b, a = scipy.signal.butter(2, [2/25,15/25], 'bandpass')
    for i in range(dataX.shape[0]):
        dataX[i]=scipy.signal.filtfilt(b,a,dataX[i],axis=0)
    dataXTest=scipy.signal.filtfilt(b,a,dataXTest,axis=1)
    tmp0 = 0
    count = 0
    resCount = 20
    dataXIn = np.zeros((1000,2000,1,3))
    dataYIn = np.zeros((1000,2000,1,1))
    p0 = 0
    rms0 = 10000
    model0 = model
    JPCount = 0
    SCCount = 0
    noUse=np.ones(N+1)
    usePhase=np.zeros(N+1)
    badY=np.zeros((1,2000))-1
    badCount=0
    badCount1=0
    for i in range(2000):
        j=0
        for jj in range(1500):
            JPCount = (JPCount+1) % N
            index = JPCount % N+eIndexTest
            noUse[index-eIndexTest] = 0
            usePhase[index-eIndexTest] = usePhase[index-eIndexTest]+1
            indexO = math.floor(np.random.rand()*N1)+N2
            theta = np.random.rand()*3.1415927
            if j==1000:
                break
            if dataX[index, (indexO+0):(indexO+2000), :,:].std(axis=0).min()==0:
                continue
            dataXIn[j, :, :, 0] = dataX[index, (indexO+0):(indexO+2000), :, 0]*cos(theta) +\
                    dataX[index, (indexO+0):(indexO+2000), :, 1]*sin(theta)
            dataXIn[j, :, :, 1] = -dataX[index, (indexO+0): \
            (indexO+2000), :, 0]*sin(theta) +\
                    dataX[index, (indexO+0):(indexO+2000), :, 1]*cos(theta)
            dataXIn[j, :, :, 2] = dataX[index, (indexO+0):(indexO+2000), :, 2]
            dataYIn[j, :, :, :] = (dataY[index, (indexO+0):(indexO+2000)]).reshape([-1, 1, 1])
            j+=1
        j=0
        for jj in range(SCN+20):
            if j==SCN:
                break
            SCCount=(SCCount+1)%(SCN0-1000)
            theta=np.random.rand()*3.1415927
            index=SCCount%(SCN0-1000)+1000
            indexO=math.floor(np.random.rand()*600)+200
            if dataXTest[index, (indexO+0):(indexO+2000)].std(axis=0).min()==0:
                continue
            dataXIn[j, :, :, 0] = dataXTest[index, (indexO+0):(indexO+2000)\
            , :, 0]*cos(theta) + \
                    dataXTest[index,(indexO+0):(indexO+2000),:,1]*sin(theta)
            dataXIn[j,:,:,1]=-dataXTest[index,(indexO+0):(indexO+2000),:,0]*sin(theta)+\
                    dataXTest[index,(indexO+0):(indexO+2000),:,1]*cos(theta)
            dataXIn[j,:,:,2] = dataXTest[index,(indexO+0):(indexO+2000),:,2]
            dataYIn[j,:,:,:] = (dataYTest[index,(indexO+0):(indexO+2000),0,0]).reshape([-1, 1, 1])
            j+=1
        for j in range(SCN,SCN+50):
            badCount1=badCount1+1
            badI=random.choice(np.arange(len(badSacDataL)))
            #print('###### badSacIndex %d'%badI)
            badSacData=badSacDataL[badI]
            badL=badLL[badI]
            badIndex=int(random.choice(badL)+np.random.rand()*3000-1500-1000)
            #print('###### badSacIndex %d %d'%(badI,badIndex))
            #badIndex=int(badL[badCount]+np.random.rand()*3000-1500-1000)
            dataXIn[j,:,:,:]=processX(badSacData[badIndex:badIndex+2000,:].reshape([1,2000,1,3]))
            if np.random.rand()<=0.1:
                iii0=int(np.random.rand()*1200+100)
                iii1=iii0+int(np.random.rand()*600)+10
                for comp in range(3):
                    dataXIn[j,iii0:iii1,:,comp]=dataXIn[j,iii0:iii1,:,comp]*0\
                    +np.random.rand()*8*dataXIn[j,iii0:iii1,:,comp].max()
            dataYIn[j,:,:,:]=badY.reshape([1,2000,1,1])
        dataXIn=processX(dataXIn,isNoise=False)
        ne =3
        if i >3:
            ne =1
        if i >20 and i%10==0:
            K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * 0.9)#0.95
        bs = 50
        if i > 5:
            bs = 75
        if i > 10:
            bs = 100
        if i > 20:
            bs = 110
        if i > 30:
            bs = 120
        if i >40:
            bs = 125
        if i > 50:
            bs = 130
        if i > 100:
            bs = 150
        if i > 150:
            bs = 200
        if i > 300:
            bs=300
        #if i==300:
        #   model.compile(optimizer='SGD')
        model.fit(dataXIn,dataYIn.reshape([-1, 2000, 1, 1]), nb_epoch=ne,  \
         batch_size=bs, verbose=2)
        logger.info('%d phases no Use %d loop'%(noUse.sum(),usePhase[10]))
        if i%1==0:
            tmpY = model.predict(processX(dataXTest[0:1000,sN1:eN1,:,:]), verbose=0)[:,:,:1]
            thresholds = [200, 100, 20]
            minYL=[0.01, 0.1,0.2,0.3,0.4,0.5]
            for threshold in thresholds:
                for minY in minYL:
                    p,m,s=validStd(tmpY, dataYTest[0:1000, sN1: eN1, 0, 0], threshold=threshold,minY=minY)
                    logger.info('SC % 3d : minY:%.2f p:%.4f m:%.4f s:%.4f'%(threshold,minY,p,m,s))
            tmpY=model.predict(processX(dataX[0:1000, sN0:eN0, :, :]), verbose=0)[:,:,:1]
            for threshold in thresholds:
                for minY in minYL:
                    p,m,s=validStd(tmpY, dataY[0:1000, sN0: eN0], threshold=threshold, minY=minY)
                    logger.info('JP % 3d : minY:%.2f p:%.4f m:%.4f s:%.4f'%(threshold,minY,p,m,s))
            p,absMean,rms=validStd(tmpY,dataY[0:1000,sN0:eN0],threshold=20, minY=0.5)
            rms=model.evaluate(x=processX(dataX[0:1000,sN0:eN0, :, :]), y=dataY[0:1000, sN0: eN0].reshape([-1, 2000, 1, 1]))
            rms-=p*100
            if rms >= rms0 and p > 0.45 and usePhase[10]>2:
                resCount = resCount-1
                if resCount == 0:
                    model.set_weights(model0)
                    logger.info('over fit ,force to stop, set to best model')
                    break
            if rms < rms0 and p > 0.45 and  usePhase[10]>2:
                resCount = 20
                rms0 = rms
                model0 = (model.get_weights())
                logger.info('find a better model')
    model.set_weights(model0)
    b, a = scipy.signal.butter(4, [2/25,15/25], 'bandpass')
    outY = model.predict(processX(dataX[sIndexTest:eIndexTest,sN0:eN0,:,:]), verbose=0)[:,:,:1]
    outYTest = model.predict(processX(dataXTest[0:1000,sN1:eN1, :, :]), verbose=0)[:,:,:1]
    minYL=[0.01, 0.1,0.2,0.3,0.4,0.5]
    for threshold in thresholds:
        for minY in minYL:
            p,m,s=validStd(outYTest,dataYTest[0:1000, sN1:eN1, 0 , 0],threshold=threshold,minY=minY)
            logger.info('test SC % 3d : minY:%.2f p:%.4f m:%.4f s:%.4f'%(threshold,minY,p,m,s))
    for threshold in thresholds:
        for minY in minYL:
            p,m,s=validStd(outY,dataY[sIndexTest:eIndexTest,sN0:eN0], threshold=threshold, minY=minY)
            logger.info('test JP % 3d : minY:%.2f p:%.4f m:%.4f s:%.4f'%(threshold,minY,p,m,s))
    #p,m,s=validStd(outYTest,dataYTest[0:1000, sN1:eN1, 0 , 0])
    #logger.info('testRate SC  p:%.4f m:%.4f s:%.4f'%(p,m,s))
    #print('testRate SC:', validStd(outYTest,dataYTest[0:1000, sN1:eN1, 0 \
    #    , 0]))
    #p,m,s=validStd(outY,dataY[sIndexTest:eIndexTest,sN0:eN0])
    #logger.info('testRate JP  p:%.4f m:%.4f s:%.4f'%(p,m,s))
    #print('testdRate JP:', validStd(outY,dataY[sIndexTest:eIndexTest,sN0:eN0]))
    #outY = model.predict(dataX[1000:2000,sN0:eN0, :, :], verbose=0)
    #outYTest = model.predict(dataXTest[0:1000,sN1:eN1, :, :], verbose=0)
    sio.savemat(resFile, {'out'+yStr :outY, 'out'+xStr: dataX[sIndexTest:eIndexTest,sN0:eN0,:,:], \
            yStr+'0': dataY[sIndexTest:eIndexTest,sN0:eN0], yStr+'0Test': dataYTest[0:1000,sN1:eN1, 0, 0], \
            'out'+yStr+'Test': outYTest, 'out'+xStr+'Test':  \
            dataXTest[0: 1000, sN1: eN1, :, :]})
    model.save(modelFile)


def validStd(tmpY,tmpY0,threshold=100, minY=0.2):
    tmpY=tmpY.reshape((-1,2000))
    tmpY0=tmpY0.reshape((-1,2000))
    maxY0=tmpY0.max(axis=1);
    validL=np.where(maxY0>0.9)[0]
    tmpY=tmpY[validL]
    tmpY0=tmpY0[validL]
    maxYIndex=tmpY0.argmax(axis=1)
    validL=np.where((maxYIndex-250)*(maxYIndex-1750)<0)[0]
    tmpY=tmpY[validL]
    tmpY0=tmpY0[validL]

    #print(validL)
    di=(tmpY.reshape([-1,2000])[:, 250:1750].argmax(axis=1)-\
            tmpY0.reshape([-1,2000])[:, 250:1750].argmax(axis=1))
    validL=np.where(np.abs(di)<threshold)[0]
    pTmp=tmpY.reshape([-1,2000])[:, 250:1750].max(axis=1)[validL]
    validLNew=np.where(pTmp>minY)[0]
    validL=validL[validLNew]
    if len(di)==0:
        return 0, 0, 0
    return np.size(validL)/np.size(di),di[validL].mean(),di[validL].std()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--phase', '-p', type=str, help='train for p or s')
    parser.add_argument('--SCN', '-n', type=int, help='number of SC phase')
    parser.add_argument('--Num', '-N', type=int, help='number of JP phase')
    args = parser.parse_args()
    SCN = 0
    N = 150000-13000
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s -\
            %(message)s')
    if args.SCN:
        SCN = args.SCN
    if args.Num:
        N = args.Num-13000
    if args.phase == 'p':
        trainP(N, SCN)
    if args.phase == 's':
        trainS(N, SCN)
