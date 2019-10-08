from keras.models import *
from keras import optimizers
from keras.layers import LSTM, Dense, TimeDistributed, GRU, MaxPooling1D, AveragePooling1D, SimpleRNN,Softmax,Activation,Dropout
from keras.layers import *
import math
import scipy.io as sio
import numpy as np
import os
from keras import backend as K

w1=np.ones(1500)*0.5
w0=np.ones(250)*(-0.75)
w2=np.ones(250)*(-0.25)
w=np.append(w0,w1)
w=np.append(w,w2)
wY=K.variable(w.reshape((1,2000,1,1)))

w11=np.ones(1500)*0
w01=np.ones(250)*(-0.75)
w21=np.ones(250)*(-0.25)
w1=np.append(w01,w11)
w1=np.append(w1,w21)
wY1=K.variable(w1.reshape((1,2000,1,1)))

def lossFunc(y,yout):
    #yNew=K.concatenate([y,1-y],axis=3)
    #youtNew=K.concatenate([yout,1-yout],axis=3)
    #return -K.mean(yNew*K.log(youtNew+1e-9),axis=[0,1,2,3])
    return -K.mean((y*K.log(yout+1e-9)+(1-y)*(K.log(1-yout+1e-9)))*(y*0+1)*(1+K.sign(y)*wY),axis=[0,1,2,3])

def lossFuncNew(y,yout):
    #to S
    #y0=K.sqrt(K.mean(y**0.25,axis=[0,1,2,3]))*1.5#3
    yW=(K.sign(-y-0.1)+1)*100*(K.sign(yout-0.35)+1)+1
    y=(K.sign(y+0.1)+1)*y/2
    y0=0.13
    #yNew=K.concatenate([y,1-y],axis=3)
    #youtNew=K.concatenate([yout,1-yout],axis=3)
    #return -K.mean(yNew*K.log(youtNew+1e-9),axis=[0,1,2,3])
    return -K.mean((y*K.log(yout+1e-9)/y0+(1-y)*(K.log(1-yout+1e-9))/(1-y0))*(y*0+1)*(1+K.sign(y)*wY1)*yW,axis=[0,1,2,3])

def lossFuncNewS(y,yout):
    y=y
    #y0=K.sqrt(K.mean(y,axis=[0,1,2,3]))*1.5
    yW=(K.sign(-y-0.1)+1)*100*(K.sign(yout-0.35)+1)+1
    y=(K.sign(y+0.1)+1)*y/2
    y0=0.13
    #yNew=K.concatenate([y,1-y],axis=3)
    #youtNew=K.concatenate([yout,1-yout],axis=3)
    #return -K.mean(yNew*K.log(youtNew+1e-9),axis=[0,1,2,3])
    return -K.mean((y*K.log(yout+1e-9)/y0+(1-y)*(K.log(1-yout+1e-9))/(1-y0))*(y*0+1)*(1+K.sign(y)*wY1)*yW,axis=[0,1,2,3])

def lossFuncSoft(y,yout):
    yNew=K.concatenate([y,1-y],axis=3)
    youtNew=yout#K.concatenate([yout,1-yout],axis=3)
    return -K.sum(K.mean(yNew*K.log(youtNew+1e-9),axis=[0,1,2])/K.mean(yNew,axis=[0,1,2]),axis=-1)
    #return -K.mean((y*K.log(yout+1e-9)+(1-y)*(K.log(1-yout+1e-9)))*(y*0+1)*(1+K.sign(y)*wY),axis=[0,1,2,3])

def genModel(phase='p'):
    in_features = 3
    n_1 = 2000
    n_2=1
    n_out = 24
    out_dim1 = 200
    out_dim2 = 50
    out_features1 = 100
    out_features2 = 100
    vectorLen=5
    filt1=25
    filt2=125
    filt3=100#100
    filt4=125
    filt5=50#100#50#75
    filt6=75#50
    filt7=100
    filt7T=100
    filterT5=75#75
    filterT4=50#50####50
    filterT3=125
    filterT2=100#100####100
    filterT1=125#75
    filterT0=25#50
    acL={1:'relu',2:'tanh',3:'relu',4:'tanh',5:'relu',6:'tanh',7:'relu'}
    #acL={1:'relu',2:'relu',3:'relu',4:'relu',5:'relu',6:'relu',7:'relu'}
    
    inputs=Input((n_1,n_2, in_features));
    conv1=Conv2D(filt1,kernel_size=(10,1) ,strides=(1,1), padding='same',activation=acL[1])(inputs)
    pool1=MaxPooling2D(pool_size=(5,1), strides=(5,1),padding='same')(conv1)
    
    conv2=Conv2D(filt2,kernel_size=(10,1) ,strides=(1,1), padding='same',activation=acL[2])(pool1)
    pool2=AveragePooling2D(pool_size=(5,1), strides=(5,1), padding='same')(conv2)
    
    conv3=Conv2D(filt3,kernel_size=(4,1) ,strides=(1,1), padding='same',activation=acL[3])(pool2)
    pool3=MaxPooling2D(pool_size=(2,1), strides=(2,1),padding='same')(conv3)
    
    conv4=Conv2D(filt4,kernel_size=(4,1) ,strides=(1,1), padding='same',activation=acL[4])(pool3)
    pool4=AveragePooling2D(pool_size=(2,1), strides=(2,1), padding='same')(conv4)
    
    
    conv5=Conv2D(filt5,kernel_size=(4,1) ,strides=(1,1), padding='same',activation=acL[5])(pool4)
    pool5=MaxPooling2D(pool_size=(2,1), strides=(2,1),padding='same')(conv5)
    
    conv6=Conv2D(filt6,kernel_size=(4,1) ,strides=(1,1), padding='same',activation=acL[6])(pool5)
    pool6=MaxPooling2D(pool_size=(2,1), strides=(2,1),padding='same')(conv6)

    dConv6=Conv2D(filt7,kernel_size=(4,1) ,strides=(1,1), padding='same',activation=acL[7])(pool6)
    dConv6M=dConv6#Conv2D(filt7T,kernel_size=(4,1) ,strides=(1,1), padding='same',activation=acL[7])(dConv6)
    
    dConv5=Conv2DTranspose(filterT5,kernel_size=(3,1),strides=(2,1),padding='same',activation=acL[6])(dConv6M)
    dConv5M=concatenate([dConv5,conv6],axis=3)
    
    dConv4=Conv2DTranspose(filterT4,kernel_size=(3,1),strides=(2,1),padding='same',activation=acL[5])(dConv5M)
    dConv4M=concatenate([dConv4,conv5],axis=3)
    
    dConv3=Conv2DTranspose(filterT3,kernel_size=(3,1),strides=(2,1),padding='same',activation=acL[4])(dConv4M)
    dConv3M=concatenate([dConv3,conv4],axis=3)
    
    dConv2=(Conv2DTranspose(filterT2,kernel_size=(3,1),strides=(2,1),padding='same',activation=acL[3])(dConv3M))
    dConv2M=concatenate([dConv2,conv3],axis=3)
    
    dConv1=(Conv2DTranspose(filterT1,kernel_size=(6,1),strides=(5,1),padding='same',activation=acL[2])(dConv2M))
    dConv1M=concatenate([dConv1,conv2],axis=3)
    
    dConv0=Conv2DTranspose(filterT0,kernel_size=(6,1),strides=(5,1),padding='same',activation=acL[1])(dConv1M)
    #relu
    dConv0M=concatenate([dConv0,conv1],axis=3)

    outputA=Conv2D(1,kernel_size=(4,1),strides=(1,1),padding='same',activation='sigmoid')(dConv0M)#,activation='sigmoid')
    #outputA=(Softmax(axis=3)(outputA0))
    model=Model(inputs=inputs,outputs=outputA)
    if phase=='p':
        model.compile(loss=lossFuncNew, optimizer='Nadam')
    else:
        model.compile(loss=lossFuncNewS, optimizer='Nadam')
    return model

def genModelSoft():
    in_features = 3
    n_1 = 2000
    n_2=1
    n_out = 24
    out_dim1 = 200
    out_dim2 = 50
    out_features1 = 100
    out_features2 = 100
    vectorLen=5
    filt1=25
    filt2=125
    filt3=100#100
    filt4=125
    filt5=50#100#50#75
    filt6=75#50
    filt7=100
    filt7T=100
    filterT5=75#75
    filterT4=50#50####50
    filterT3=125
    filterT2=100#100####100
    filterT1=125#75
    filterT0=25#50
    acL={1:'relu',2:'tanh',3:'relu',4:'tanh',5:'relu',6:'tanh',7:'relu'}
    #acL={1:'relu',2:'relu',3:'relu',4:'relu',5:'relu',6:'relu',7:'relu'}
    
    inputs=Input((n_1,n_2, in_features));
    conv1=Conv2D(filt1,kernel_size=(10,1) ,strides=(1,1), padding='same',activation=acL[1])(inputs)
    pool1=MaxPooling2D(pool_size=(5,1), strides=(5,1),padding='same')(conv1)
    
    conv2=Conv2D(filt2,kernel_size=(10,1) ,strides=(1,1), padding='same',activation=acL[2])(pool1)
    pool2=AveragePooling2D(pool_size=(5,1), strides=(5,1), padding='same')(conv2)
    
    conv3=Conv2D(filt3,kernel_size=(4,1) ,strides=(1,1), padding='same',activation=acL[3])(pool2)
    pool3=MaxPooling2D(pool_size=(2,1), strides=(2,1),padding='same')(conv3)
    
    conv4=Conv2D(filt4,kernel_size=(4,1) ,strides=(1,1), padding='same',activation=acL[4])(pool3)
    pool4=AveragePooling2D(pool_size=(2,1), strides=(2,1), padding='same')(conv4)
    
    
    conv5=Conv2D(filt5,kernel_size=(4,1) ,strides=(1,1), padding='same',activation=acL[5])(pool4)
    pool5=MaxPooling2D(pool_size=(2,1), strides=(2,1),padding='same')(conv5)
    
    conv6=Conv2D(filt6,kernel_size=(4,1) ,strides=(1,1), padding='same',activation=acL[6])(pool5)
    pool6=MaxPooling2D(pool_size=(2,1), strides=(2,1),padding='same')(conv6)

    dConv6=Conv2D(filt7,kernel_size=(4,1) ,strides=(1,1), padding='same',activation=acL[7])(pool6)
    dConv6M=dConv6#Conv2D(filt7T,kernel_size=(4,1) ,strides=(1,1), padding='same',activation=acL[7])(dConv6)
    
    dConv5=Conv2DTranspose(filterT5,kernel_size=(3,1),strides=(2,1),padding='same',activation=acL[6])(dConv6M)
    dConv5M=concatenate([dConv5,conv6],axis=3)
    
    dConv4=Conv2DTranspose(filterT4,kernel_size=(3,1),strides=(2,1),padding='same',activation=acL[5])(dConv5M)
    dConv4M=concatenate([dConv4,conv5],axis=3)
    
    dConv3=Conv2DTranspose(filterT3,kernel_size=(3,1),strides=(2,1),padding='same',activation=acL[4])(dConv4M)
    dConv3M=concatenate([dConv3,conv4],axis=3)
    
    dConv2=(Conv2DTranspose(filterT2,kernel_size=(3,1),strides=(2,1),padding='same',activation=acL[3])(dConv3M))
    dConv2M=concatenate([dConv2,conv3],axis=3)
    
    dConv1=(Conv2DTranspose(filterT1,kernel_size=(6,1),strides=(5,1),padding='same',activation=acL[2])(dConv2M))
    dConv1M=concatenate([dConv1,conv2],axis=3)
    
    dConv0=Conv2DTranspose(filterT0,kernel_size=(6,1),strides=(5,1),padding='same',activation=acL[1])(dConv1M)
    #relu
    dConv0M=concatenate([dConv0,conv1],axis=3)

    outputA0=Conv2D(2,kernel_size=(4,1),strides=(1,1),padding='same',activation='relu')(dConv0M)#,activation='sigmoid')
    outputA=Softmax(axis=3)(outputA0)
    model=Model(inputs=inputs,outputs=outputA)
    model.compile(loss=lossFuncSoft, optimizer='adam')
    return model
